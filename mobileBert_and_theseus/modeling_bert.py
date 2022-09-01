"""PyTorch BERT model. """
import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
# import tensorflow as tf
import torch.nn.functional as F

from .switchable_norm import SwitchNorm1d
from transformers.activations import gelu, gelu_new, swish
from transformers.configuration_bert import BertConfig
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_callable,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from transformers.utils import logging


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "bert-base-german-dbmdz-cased",
    "bert-base-german-dbmdz-uncased",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]


# Load tf checkpoints in a pytorch model.
def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
                     "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    count = 0
    for name, array in zip(names, arrays):
        name = name.replace("ffn_layer", "ffn")
        name = name.replace("cls/predictions/transform/LayerNorm", "cls/predictions/transform/FakeLayerNorm")
        name = name.replace("extra_output_weights", 'dense/kernel')

        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue

        if count <= 1128:
            pointer = model
            for m_name in name:
                if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                    scope_names = re.split(r"_(\d+)", m_name)
                else:
                    scope_names = [m_name]

                if scope_names[0] == "kernel" or scope_names[0] == "gamma":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                    pointer = getattr(pointer, "bias")
                elif scope_names[0] == "output_weights":
                    pointer = getattr(pointer, "weight")
                elif scope_names[0] == "squad":
                    pointer = getattr(pointer, "classifier")
                else:
                    try:
                        pointer = getattr(pointer, scope_names[0])
                    except AttributeError:
                        logger.info("Skipping {}".format("/".join(name)))
                        continue
                if len(scope_names) >= 2:
                    num = int(scope_names[1])
                    pointer = pointer[num]

            if m_name[-11:] == "_embeddings":
                pointer = getattr(pointer, "weight")
            elif m_name == "kernel":
                array = np.transpose(array)
            try:
                assert (pointer.shape == array.shape), f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            logger.info("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
            count = count + 1
        else:
            break
    return model


# ------------------------------------------Embeddings------------------------------------------ #
# ---------------------------------------------------------------------------------------------- #
def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}


class FakeBertLayerNorm(nn.Module):
    # Construct a layernorm module in the TF style (epsilon inside the square root).
    def __init__(self, parameter, eps=1e-12):
        super(FakeBertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(parameter))
        self.bias = nn.Parameter(torch.zeros(parameter))
        self.variance_epsilon = eps

    def forward(self, x):
        # u = x.mean(-1, keepdim=True)
        # s = (x - u).pow(2).mean(-1, keepdim=True)
        # x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# Construct the embeddings from word, position and token_type embeddings.
class BertEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.embedding_transformation = nn.Linear(config.embedding_size*3, config.hidden_size)
        self.FakeLayerNorm = FakeBertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.config = config
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # input_id shape
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        # get sequence length
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.config.trigram_input:
            inputs = inputs_embeds
            inputs1 = F.pad(inputs[:, 1:], pad=(0, 0, 0, 1, 0, 0), mode="constant", value=0.0)
            inputs2 = F.pad(inputs[:, :-1], pad=(0, 0, 1, 0, 0, 0), mode="constant", value=0.0)
            inputs_embeds = torch.cat((inputs1,inputs,inputs2),2)
        else:
            inputs_embeds = inputs_embeds
        inputs_embeds = self.embedding_transformation(inputs_embeds)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings

        # embeddings (batch_size, seq_len, hidden_size)
        embeddings = self.FakeLayerNorm(embeddings)
        # embeddings = FakeLayerNorm(embeddings)
        # embeddings= self.intermediate_act_fn(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# --------------------------------------------BERT---------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# - - - - - -  - - - -  -  - -  - -  - - -  - -  - - -  - - - - - - - - - - - - -   - - - -  - - #
class BottleneckShrinkInput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intra_bottleneck_size)
        self.FakeLayerNorm = FakeBertLayerNorm(config.intra_bottleneck_size, eps=config.layer_norm_eps)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        layer_input = self.dense(hidden_states)
        # layer_input (batch_size, seq_len, intra_bottleneck_size)
        layer_input = self.FakeLayerNorm(layer_input)
        # layer_input = FakeLayerNorm(layer_input)
        # layer_input = self.intermediate_act_fn(layer_input)
        return layer_input


class BottleneckShrinkAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intra_bottleneck_size)
        self.FakeLayerNorm = FakeBertLayerNorm(config.intra_bottleneck_size, eps=config.layer_norm_eps)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        shared_attention_input = self.dense(hidden_states)
        # layer_input (batch_size, seq_len, intra_bottleneck_size)
        shared_attention_input = self.FakeLayerNorm(shared_attention_input)
        # shared_attention_input = FakeLayerNorm(shared_attention_input)
        # shared_attention_input = self.intermediate_act_fn(shared_attention_input)
        return shared_attention_input


class BottleneckShrink(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input = BottleneckShrinkInput(config)
        self.attention = BottleneckShrinkAttention(config)

    def forward(self, hidden_states, attention_mask=None):
        layer_input = self.input(hidden_states)
        shared_attention_input = self.attention(hidden_states)
        query_tensor = shared_attention_input
        key_tensor = shared_attention_input
        value_tensor = hidden_states
        return layer_input, query_tensor, key_tensor, value_tensor
        # return shared_attention_input, query_tensor, key_tensor, value_tensor


# - - - - - -  - - - -  -  - -  - -  - - -  - -  - - -  - - - - - - - - - - - - -   - - - -  - - #
class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.intra_bottleneck_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" %
                             (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.intra_bottleneck_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.intra_bottleneck_size, self.all_head_size)
        self.key = nn.Linear(config.intra_bottleneck_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, query_tensor, key_tensor, value_tensor, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False,):
        mixed_query_layer = self.query(query_tensor)
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(key_tensor)
            mixed_value_layer = self.value(value_tensor)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(key_tensor)
            mixed_value_layer = self.value(value_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(float(self.attention_head_size))
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intra_bottleneck_size, config.intra_bottleneck_size)
        self.FakeLayerNorm = FakeBertLayerNorm(config.intra_bottleneck_size, eps=config.layer_norm_eps)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        # hidden_states = self.dropout(hidden_states)
        # hidden_states (batch_size, seq_len, intra_bottleneck_size)
        hidden_states = self.FakeLayerNorm(hidden_states + input_tensor)
        # hidden_states = FakeLayerNorm(hidden_states + input_tensor)
        # hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, layer_input, query_tensor, key_tensor, value_tensor, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False,):
        self_outputs = self.self(query_tensor, key_tensor, value_tensor, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions,)
        attention_output = self.output(self_outputs[0], layer_input)
        # attention_output = self.intermediate_act_fn(attention_output)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# - - - - - -  - - - -  -  - -  - -  - - -  - -  - - -  - - - - - - - - - - - - -   - - - -  - - #
class FFN_layerintermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intra_bottleneck_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, attention_output):
        intermediate_output = self.dense(attention_output)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        return intermediate_output


class FFN_layeroutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intra_bottleneck_size)
        self.FakeLayerNorm = FakeBertLayerNorm(config.intra_bottleneck_size, eps=config.layer_norm_eps)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, intermediate_output, attention_output):
        intermediate_output = self.dense(intermediate_output)
        # intermediate_output (batch_size, seq_len, intra_bottleneck_size)
        layer_output = self.FakeLayerNorm(intermediate_output+attention_output)
        # layer_output = self.FakeLayerNorm(intermediate_output)
        # layer_output = FakeLayerNorm(intermediate_output+attention_output)
        # layer_output = self.intermediate_act_fn(layer_output)
        return layer_output


class FFN_layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate = FFN_layerintermediate(config)
        self.output = FFN_layeroutput(config)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        # layer_output = self.intermediate_act_fn(layer_output)
        return layer_output


# - - - - - -  - - - -  -  - -  - -  - - -  - -  - - -  - - - - - - - - - - - - -   - - - -  - - #
class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intra_bottleneck_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# - - - - - -  - - - -  -  - -  - -  - - -  - -  - - -  - - - - - - - - - - - - -   - - - -  - - #
class Bottleneckoutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intra_bottleneck_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.FakeLayerNorm = FakeBertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, input_tensor):
        layer_output = self.dense(hidden_states)
        layer_output = self.dropout(layer_output)
        layer_output = self.FakeLayerNorm(layer_output+input_tensor)
        # layer_output = FakeLayerNorm(layer_output+input_tensor)
        # layer_output = self.intermediate_act_fn(layer_output)
        return layer_output


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.intra_bottleneck_size)
        self.FakeLayerNorm = FakeBertLayerNorm(config.intra_bottleneck_size, eps=config.layer_norm_eps)
        self.bottleneck = Bottleneckoutput(config)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states, input_tensor, prev_output):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.FakeLayerNorm(hidden_states + input_tensor)
        # hidden_states = self.FakeLayerNorm(hidden_states)
        # hidden_states = FakeLayerNorm(hidden_states + input_tensor)
        # hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.bottleneck(hidden_states, prev_output)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bottleneck = BottleneckShrink(config)
        self.attention = BertAttention(config)
        self.ffn = nn.ModuleList([FFN_layer(config) for _ in range(config.num_feedforward_networks-1)])
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=False,):
        layer_input, query_tensor, key_tensor, value_tensor = self.bottleneck(hidden_states, attention_mask)
        self_attention_outputs = self.attention(layer_input, query_tensor, key_tensor, value_tensor, attention_mask, head_mask, output_attentions=output_attentions,)

        layer_input = self_attention_outputs[0]
        for i, layer_module in enumerate(self.ffn):
            layer_output = layer_module(layer_input)
            layer_input = layer_output

        intermediate_output = self.intermediate(layer_input)
        layer_output = self.output(intermediate_output, layer_input, hidden_states)
        return layer_output


# --------------------------------------------Pooler-------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.classifier_activation = config.classifier_activation
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the mo b del by simply taking the hidden state corresponding to the first token.
        first_token_tensor = hidden_states[:, 0]
        if self.classifier_activation:
            pooled_output = self.dense(first_token_tensor)
            pooled_output = self.activation(pooled_output)
            return pooled_output
        else:
            pooled_output = first_token_tensor
            return pooled_output


# --------------------------------------Pretreatment-------------------------------------------- #
# ---------------------------------------------------------------------------------------------- #
# - - - -  - - - - - - -- - - - - - - - - - - - - -- - - - - - - -  - -- - - - - - - - - - - - - #
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.FakeLayerNorm = FakeBertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.FakeLayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is an output-only bias for each token.
        self.dense = nn.Linear(config.vocab_size, config.hidden_size - config.embedding_size, bias=False)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = hidden_states.matmul(torch.cat([self.decoder.weight.t(), self.dense.weight], dim=1))
        hidden_states += self.bias
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


# ----------------------------------------------------------------------------------------------------- #
BERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.__call__` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`{0}`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the hidden states of all layers are returned. See ``hidden_states`` under returned tensors for more detail.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.ModelOutput` instead of a
            plain tuple.
"""


class BertPreTrainedModel(PreTrainedModel):
    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # elif isinstance(module, BertLayerNorm):
        #     module.bias.data.zero_()
        #     module.weight.data.fill_(1.0)
        elif isinstance(module, FakeBertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states,attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                output_attentions=False, output_hidden_states=False, return_dict=False,):
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(self.config, "gradient_checkpointing", False):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class BertModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint="bert-base-uncased",
                                output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC,)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None,):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# - - - - - - - - - -- --  -- - -- - - --  -- - - -- - - - - - -- - -- -- - - - - -- - - - #
@dataclass
class BertForPreTrainingOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@add_start_docstrings(
    """Bert Model with two heads on top as done during the pre-training: a `masked language modeling` head and
    a `next sentence prediction (classification)` head. """,
    BERT_START_DOCSTRING,
)
class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    @replace_return_docstrings(output_type=BertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, next_sentence_label=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        r"""
            labels (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`, defaults to :obj:`None`):
                Labels for computing the masked language modeling loss.
                Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
                Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
                in ``[0, ..., config.vocab_size]``
            next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`, defaults to :obj:`None`):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see :obj:`input_ids` docstring)
                Indices should be in ``[0, 1]``.
                ``0`` indicates sequence B is a continuation of sequence A,
                ``1`` indicates sequence B is a random sequence.
            kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
                Used to hide legacy arguments that have been deprecated.

        Returns:
        """

        if "masked_lm_labels" in kwargs:
            warnings.warn("The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.", FutureWarning,)
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask,
                            inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict,)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(loss=total_loss, prediction_logits=prediction_scores, seq_relationship_logits=seq_relationship_score,
                                        hidden_states=outputs.hidden_states, attentions=outputs.attentions,)


# --------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------- #
class BertLMPredictionHead2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)
        self.linear = nn.Linear(config.hidden_size, config.embedding_size, bias=False)
        # The output weights are the same as the input embeddings, but there is an output-only bias for each token.
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.linear(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead2(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


# Returns a list of the shape of tensor, preferring static dimensions.
def get_shape_list(tensor, expected_rank=None, name=None):
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape
  
  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape

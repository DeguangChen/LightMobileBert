import tqdm
import numpy as np
import torch
import tensorly as tl

tl.set_backend("pytorch")
from tensorly.decomposition import parafac
from tensorly.decomposition import tucker

from .collaborative_attention import CollaborativeAttention, MixingMatrixInit


# def pca(X, k):  # k is the components you want
#     # mean of each feature
#     n_samples, n_features = X.shape
#     mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
#     # normalization
#     norm_X = X - mean
#     # scatter matrix
#     scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
#     # Calculate the eigenvectors and eigenvalues
#     eig_val, eig_vec = np.linalg.eig(scatter_matrix)
#     eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
#     # sort eig_vec based on eig_val from highest to lowest
#     eig_pairs.sort(reverse=True)
#     # select the top k eig_vec
#     feature = np.array([ele[1] for ele in eig_pairs[:k]])
#     # get new data
#     data = np.dot(norm_X, np.transpose(feature))
#     return data


def swap_to_collaborative(model, adapter, dim_shared_query_key, tol=1e-6):
    print("Swap concatenate self-attention layers to collaborative...")
    for i in tqdm.trange(adapter.num_layers(model)):
        # plug the current layer into the adapter to have access to the fields we need
        a = adapter.get_layer(model, i)
        layer = adapter(a)

        # create the collaborative layer
        new_layer = CollaborativeAttention(
            dim_input=layer.dim_input,
            dim_value_all=layer.dim_value_all,
            value_dim_input=layer.value_dim_input,
            dim_key_query_all=dim_shared_query_key,
            dim_output=layer.dim_output,
            num_attention_heads=layer.num_attention_heads,
            output_attentions=False,
            attention_probs_dropout_prob=layer.attention_probs_dropout_prob,
            use_dense_layer=layer.use_dense_layer,
            use_layer_norm=layer.use_layer_norm,
            # use_layer_norm=True,
            mixing_initialization=MixingMatrixInit.CONCATENATE,
        )

        WK_per_head = layer.WK.view([layer.num_attention_heads, -1, layer.dim_input])
        WK_per_bias = layer.bK.view([2, -1])

        if layer.dim_key_query_all != dim_shared_query_key:
            # tensor decomposition to get shared projections and mixing
            WQ_per_head = layer.WQ.view([layer.num_attention_heads, -1, layer.dim_input])
            WQ_per_bias = layer.bQ.view([2, -1])
            WQWKT_per_head = torch.einsum("hdq,hdk->qhk", WQ_per_head, WK_per_head)
            # WQWKT_per_bias = torch.einsum("hq,hk->qhk", WQ_per_bias, WK_per_bias)

            # tensor decomposition
            # ## CP decomposition
            _, factors = parafac(WQWKT_per_head.detach(), dim_shared_query_key, init="random", tol=tol)
            # ## tucker decomposition
            # cores, factors = tucker(WQWKT_per_head.detach(), dim_shared_query_key)
            WQ_shared, mixing, WK_shared = factors
            # WQ_shared_bias, mixing_bias, WK_shared_bias = factors_bias
            new_layer.key.weight.data.copy_(WK_shared.transpose(0, 1))
            new_layer.query.weight.data.copy_(WQ_shared.transpose(0, 1))
            # new_layer.key.bias.data.copy_(WK_shared_bias.squeeze(1))
            # new_layer.query.bias.data.copy_(WQ_shared_bias.squeeze(1))
            new_layer.mixing.data.copy_(mixing)
        else:
            # we simply copy the original matrices, mixing is initialized to concatenate
            new_layer.key.weight.data.copy_(layer.WK)
            new_layer.query.weight.data.copy_(layer.WQ)

        # bias reparametrization
        bq_per_head = layer.bQ.reshape([layer.num_attention_heads, -1])
        # print(bq_per_head.shape)
        # print(bq_per_head.unsqueeze(1).shape)
        content_bias = bq_per_head.unsqueeze(1) @ WK_per_head
        content_bias = content_bias.squeeze(1)
        # print(content_bias.shape)
        new_layer.content_bias.weight.data.copy_(content_bias)

        # value parameters are simply copied
        new_layer.value.weight.data.copy_(layer.WV)
        new_layer.value.bias.data.copy_(layer.bV)

        # copy output dense layer if exists
        if layer.use_dense_layer:
            new_layer.dense.weight.data.copy_(layer.WO)
            new_layer.dense.bias.data.copy_(layer.bO)

        # copy layernorm if exists
        if layer.use_layer_norm:
            new_layer.layer_norm.weight.data.copy_(layer.layerNorm.weight)
            new_layer.layer_norm.bias.data.copy_(layer.layerNorm.bias)

        new_layer = new_layer.to(layer.device)
        new_layer = adapter.wrap_layer_args(new_layer)
        adapter.set_layer(model, i, new_layer)



# coding=utf-8
"""
    Convert BERT checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import numpy as np
import argparse
import tensorflow as tf
import torch

#from modeling import BertConfig, BertForPreTraining, load_tf_weights_in_bert
from mobileBert_and_theseus.modeling_bert import BertConfig,  BertForPreTraining, load_tf_weights_in_bert


# convert tf_checkpoint to pytorch
def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_path):

    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))

    # load model
    model = BertForPreTraining(config)

    load_tf_weights_in_bert(model, config, tf_checkpoint_path)
    print("Save PyTorch model to {}".format(pytorch_dump_path))

    # save pytorch
    torch.save(model.state_dict(), pytorch_dump_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # # Required parameters
    parser.add_argument("--tf_checkpoint_path",
                        default="./uncased_L-24_H-128_B-512_A-4_F-4_OPT/mobilebert_variables.ckpt",
                        type=str,
                        help="Path the TensorFlow checkpoint path.")
    parser.add_argument("--bert_config_file",
                        default="./uncased_L-24_H-128_B-512_A-4_F-4_OPT/config.json",
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model.This specifies the model architecture.")
    parser.add_argument("--pytorch_dump_path",
                        default="./uncased_L-24_H-128_B-512_A-4_F-4_OPT/pytorch_model.bin",
                        type=str,
                        help="Path to the output PyTorch model.")

    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.bert_config_file, args.pytorch_dump_path)









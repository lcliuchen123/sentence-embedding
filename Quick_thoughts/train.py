# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train the skip-thoughts model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import os
import time
import sys

sys.path.append('.')
sys.path.append('..')

# from src import configuration
# from src import s2v_model
import configuration
import s2v_model

# FLAGS = tf.flags.FLAGS
#
# # 训练时共需要修改4个参数：input_file_pattern，train_dir，word2vec_path，size
# # 需要修改
# tf.flags.DEFINE_string("input_file_pattern", '../output/sent_word_n/train-?????-of-00010',
#                        "File pattern of sharded TFRecord files containing")
# # 需要修改
# tf.flags.DEFINE_string("train_dir", '../model/train/sent_word_n/',
#                        "Directory for saving and loading checkpoints.")
#
# tf.flags.DEFINE_integer("batch_size", 128, "Batch size")
# tf.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate")
# tf.flags.DEFINE_float("clip_gradient_norm", 5.0, "Gradient clipping norm")
# tf.flags.DEFINE_integer("save_model_secs", 600, "Checkpointing frequency")  # ???????
# tf.flags.DEFINE_integer("save_summaries_secs", 600, "Summary frequency")  # ??????????
# tf.flags.DEFINE_integer("nepochs", 1, "Number of epochs")
# tf.flags.DEFINE_integer("num_train_inst", 800000, "Number of training instances")  # ???????训练集的句子数量
# tf.flags.DEFINE_string("model_config", './train.json', "Model configuration json")
# tf.flags.DEFINE_integer("max_ckpts", 5, "Max number of ckpts to keep")  # ??????????保存最近的5个模型
# # 需要修改
# tf.flags.DEFINE_string("word2vec_path", '../data/sent_word_n/', "Path to word2vec dictionary")
#
#

# tf.flags.DEFINE_integer("context_size", 1, "Prediction context size")
# tf.flags.DEFINE_boolean("dropout", False, "Use dropout")
# tf.flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")
# tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
# tf.flags.DEFINE_boolean("shuffle_input_data", False, "Whether to shuffle data")
# tf.flags.DEFINE_integer("input_queue_capacity", 640000, "Input data queue capacity")
# tf.flags.DEFINE_integer("num_input_reader_threads", 1, "Input data reader threads")
#
# # 该参数不知道用在哪？？？？？？？？？？？？？？？？？？？？
# tf.flags.DEFINE_integer("learning_rate_decay_steps", 40000, "Learning rate decay steps")  # ？？？？？？？？
# tf.flags.DEFINE_integer("sequence_length", 30, "Max sentence length considered") # 在训练时不起作用


this_file_path = os.path.split(os.path.realpath(__file__))[0]
tf.logging.set_verbosity(tf.logging.INFO)


def main(input_file_pattern, train_dir, model_config, word2vec_path,
         learning_rate=0.005, clip_gradient_norm=5.0, uniform_init_scale=0.1,
         shuffle_input_data=False, input_queue_capacity=640000, num_input_reader_threads=1,
         dropout=False, dropout_rate=0.3, context_size=1, num_train_inst=800000,
         batch_size=128, nepochs=1, max_ckpts=5, save_summaries_secs=600, save_model_secs=600):

    start = time.time()
    if not input_file_pattern:
        raise ValueError("--input_file_pattern is required.")
    if not train_dir:
        raise ValueError("--train_dir is required.")

    with open(model_config) as json_config_file:
        model_config = json.load(json_config_file)

    model_config = configuration.model_config(model_config, mode="train", word2vec_path=word2vec_path)
    tf.logging.info("Building training graph.")
    g = tf.Graph()
    with g.as_default():
        model = s2v_model.s2v(model_config, uniform_init_scale,
                              input_file_pattern, shuffle_input_data, input_queue_capacity,
                              num_input_reader_threads, batch_size, dropout, dropout_rate, context_size,
                              mode="train")
        model.build()
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_tensor = tf.contrib.slim.learning.create_train_op(
            total_loss=model.total_loss,
            optimizer=optimizer,
            clip_gradient_norm=clip_gradient_norm)

        if max_ckpts != 5:
            saver = tf.train.Saver(max_to_keep=max_ckpts)

        else:
            saver = tf.train.Saver()

    load_words = model.init  # ????????初始化的【encode，encode】，如果fixed
    # print("load_words",load_words)
    if load_words:
        def InitAssignFn(sess):
            sess.run(load_words[0], {load_words[1]: load_words[2]})

    nsteps = int(nepochs * (num_train_inst / batch_size))

    tf.contrib.slim.learning.train(
        train_op=train_tensor,
        logdir=train_dir,
        graph=g,
        number_of_steps=nsteps,
        save_summaries_secs=save_summaries_secs,
        saver=saver,
        save_interval_secs=save_model_secs,
        init_fn=InitAssignFn if load_words else None
    )
    end = time.time()
    cost_time = end - start
    tf.logging.info("the cost time of training is %f ! ", cost_time)


if __name__ == "__main__":
    # 只需要更改前四个参数，后面的都不需要改
    input_file_pattern = '../output/second_data/new_sent_char_rem/train-?????-of-00010'
    train_dir = '../model/train/second_train/new_sent_char_rem/'
    model_config = './char_train.json'
    word2vec_path = '../data/new_sent_char_rem/'

    # num_train_inst = 800000
    # batch_size = 128
    # context_size = 1
    # nepochs = 1
    # learning_rate = 0.005
    # clip_gradient_norm = 5.0
    # uniform_init_scale = 0.1
    # shuffle_input_data = False
    # input_queue_capacity = 640000
    # num_input_reader_threads = 1
    # dropout = False
    # dropout_rate = 0.3
    # max_ckpts = 5
    # save_summaries_secs = 600
    # save_model_secs = 600

    main(input_file_pattern, train_dir, model_config, word2vec_path)

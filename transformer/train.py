
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json
import os
import time
import sys
import shutil
sys.path.append('.')
sys.path.append('..')

from transformer.model import *

this_file_path = os.path.split(os.path.realpath(__file__))[0]
tf.logging.set_verbosity(tf.logging.INFO)


# 原论文中的学习率公式进行了修改
def noam_scheme(dim, global_step, lr=0.005, warmup_steps=4000.):
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step, dtype=tf.float32)
    return lr * dim ** 0.5 * tf.minimum((step + 1.0) * warmup_steps ** -1.5, (step + 1.0) ** -0.5)


#  tensorflow2.0里面的东西
# class CustomSchedule(tf.train.optimizers.LearningRateSchedule):
#     def __init__(self, d_model, warmup_steps=4000):
#         super(CustomSchedule, self).__init__()
#
#         self.d_model = d_model
#         self.d_model = tf.cast(self.d_model, tf.float32)
#
#         self.warmup_steps = warmup_steps
#
#     def __call__(self, step):
#         arg1 = tf.math.rsqrt(step)
#         arg2 = step * (self.warmup_steps ** -1.5)
#
#         return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def main(mode, batch_size, input_file_pattern, train_dir, emb_file, num_head, num_units_list,
         vocab_size=9592, uniform_init=0.1, num_input_reader_threads=1, input_queue_capacity=640000,
         clip_gradient_norm=5, max_ckpts=5, nepochs=1, num_train_inst=800000,
         save_summaries_secs=600, save_model_secs=600):
        start = time.time()
        if not input_file_pattern and mode == 'train':
            raise ValueError("--input_file_pattern is required!!!--")
        if not train_dir:
            raise ValueError("--train_dir is required!!!--")
        tf.logging.info("Building the graph")
        g = tf.Graph()
        with g.as_default():
            model = s2v_model(mode, batch_size, input_file_pattern,
                              uniform_init, emb_file, num_head, num_units_list,
                              num_input_reader_threads, input_queue_capacity, vocab_size)
            tf.logging.info("the seq_length is %d", model.seq_length)
            model.build_loss()

            global_step = tf.train.get_or_create_global_step()
            tf.logging.info("*********the global step *************")
            tf.logging.info(global_step)
            lr = noam_scheme(model.encode_dim, global_step)
            optimizer = tf.train.AdamOptimizer(lr)
            # train_tensor = optimizer.minimize(model.total_loss, global_step=global_step)

            # optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
            train_tensor = tf.contrib.slim.learning.create_train_op(total_loss=model.total_loss,
                                                                    optimizer=optimizer,
                                                                    clip_gradient_norm=clip_gradient_norm)

            if max_ckpts != 5:
                saver = tf.train.Saver(max_to_keep=max_ckpts)
            else:
                saver = tf.train.Saver()

            load_words = model.init
            if load_words:
                def InitAssignFn(sess): #????????
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
            init_fn=InitAssignFn if load_words else None  # 初始化操作，如果为空，则调用tf.global_variables_initializer()初始化
        )
        end = time.time()
        cost_time = end - start
        tf.logging.info("the cost time of training is %f ! ", cost_time)


if __name__ == "__main__":
    input_file_pattern = '../output/transformer/sent_word_rem/train-?????-of-00010'
    train_dir = '../model/train/sent_word_rem/transformer/'
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)

    emb_file = None
    # emb_file = '../data/sent_char_n/emb.txt'
    mode = 'train'
    batch_size = 128
    num_head = 6
    tf.logging.info("************the num_head is %d*********", num_head)
    num_units_list = [1200, 300]
    main(mode=mode, batch_size=batch_size, input_file_pattern=input_file_pattern,
         train_dir=train_dir, emb_file=emb_file, num_head=num_head, num_units_list=num_units_list)

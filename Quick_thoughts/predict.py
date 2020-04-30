from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os.path
import scipy.spatial.distance as sd
import sys

sys.path.append('.')
sys.path.append('..')
# import configuration
# import encoder_manager
from Quick_thoughts import configuration, encoder_manager
import json
import tensorflow as tf
import numpy as np
from run_time import *
from process.get_predata import *

# # 生成句向量的文件必须是已经用空格隔开的
# tf.flags.DEFINE_string("encoder_file", ['../data/sent_word_n/sentences.txt'], "预处理的句子列表.")


@cost_time
def get_result(sentences, word2vec_path, model_config, input_file_pattern,
               output_dir=None, batch_size=128, context_size=1, uniform_init_scale=0.1, shuffle_input_data=False,
               input_queue_capacity=640000, num_input_reader_threads=1, dropout=False, dropout_rate=0.3):
    with open(model_config) as json_config_file:
        model_config = json.load(json_config_file)

    model_config = configuration.model_config(model_config, mode="encode", word2vec_path=word2vec_path)
    encoder = encoder_manager.EncoderManager()
    encoder.load_model(model_config, uniform_init_scale,
                       input_file_pattern, shuffle_input_data, input_queue_capacity,
                       num_input_reader_threads, batch_size, dropout, dropout_rate, context_size)

    encodings = encoder.encode(sentences)
    encodings = np.array(encodings)
    tf.logging.info(encodings)

    # output_name = os.path.join(output_dir, 'result.txt')
    # np.savetxt(output_name, encodings, fmt='%f')

    return encodings


if __name__ == "__main__":
    word2vec_path = '../data/sent_char_n/'
    model_config = './train.json'
    input_file_pattern = '../output/sent_char_n/train-?????-of-00010'
    # output_dir = '../output/sent_char_n/' # 该参数可以选择保存句向量文件的目录

    vocab = get_pre_vocab('../data/label.txt')  # label.txt是标签文件
    # 待预测的候选数据集
    file_name = '../data/人工标注数据集.txt'
    cand_sent, cand_label = get_pre_data(file_name, vocab, stop_file_name='../data/stoplist.txt')
    sen2vec = get_result(cand_sent, word2vec_path, model_config, input_file_pattern)
    print(sen2vec.shape)

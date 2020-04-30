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

import tensorflow as tf
import numpy as np
import collections
import logging
import sys
sys.path.append('.')
sys.path.append('..')
from Quick_thoughts import input_ops

# FLAGS = tf.flags.FLAGS
# tf.flags.DEFINE_integer("sequence_length", 30, "Max sentence length considered")
# tf.flags.DEFINE_integer("context_size", 1, "Prediction context size")
# tf.flags.DEFINE_boolean("dropout", False, "Use dropout")
# tf.flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")
# tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
# tf.flags.DEFINE_boolean("shuffle_input_data", False, "Whether to shuffle data")
# tf.flags.DEFINE_integer("input_queue_capacity", 640000, "Input data queue capacity")
# tf.flags.DEFINE_integer("num_input_reader_threads", 1, "Input data reader threads")


def read_vocab_embs(vocabulary_file, embedding_matrix_file):
    """生成一个单词和矩阵一一对应的词典
  :param vocabulary_file: 词典文件名，每个词典顺序与词向量文件一致
  :param embedding_matrix_file: 词向量文件名，只包含词向量，不包括对应的词
  :return:
  """
    tf.logging.info("Reading vocabulary from %s", vocabulary_file)
    with tf.gfile.GFile(vocabulary_file, mode="r") as f:
        lines = list(f.readlines())
    vocab = [line.decode("utf-8").strip() for line in lines]

    with open(embedding_matrix_file, "r") as f:
        embedding_matrix = np.load(f)
    tf.logging.info("Loaded embedding matrix with shape %s",
                    embedding_matrix.shape)
    word_embedding_dict = collections.OrderedDict(
        zip(vocab, embedding_matrix))
    return word_embedding_dict


# 该函数可以代替上面的函数
# def read_vocab_embs(embedding_matrix_file,word_dim):
#   """
#   .vector文件第一行是词的数量和维度
#   :param embedding_matrix_file: .vector词向量文件
#   :param word_dim: 词向量维度
#   :return: 词典，每个词及其对应的词向量
#   """
#   word_emb_dict = {}
#   with open(embedding_matrix_file,'r',encoding='utf-8') as f:
#     for line in f:
#       line = line.strip().split()
#       if len(line) == word_dim+1:
#         word = line[0]
#         word2vec = [float(i.strip()) for i in line[1:]]
#         if word not in word_emb_dict:
#           word_emb_dict[word] = word2vec
#
#   return word_emb_dict

def read_vocab(vocabulary_file):
    """返回一个词典：key为词，value为索引"""
    tf.logging.info("Reading vocabulary from %s", vocabulary_file)
    with tf.gfile.GFile(vocabulary_file, mode="r") as f:
        lines = list(f.readlines())
    reverse_vocab = [line.strip() for line in lines]
    print("len(reverse_vocab)", len(reverse_vocab))
    tf.logging.info("Loaded vocabulary with %d words.", len(reverse_vocab))

    # tf.logging.info("Loading embedding matrix from %s", embedding_matrix_file)
    # Note: tf.gfile.GFile doesn't work here because np.load() calls f.seek()
    # with 3 arguments.
    word_embedding_dict = collections.OrderedDict(
        zip(reverse_vocab, range(len(reverse_vocab))))
    return word_embedding_dict


class s2v(object):
    """Skip-thoughts model."""

    def __init__(self, config, uniform_init_scale, input_file_pattern, shuffle_input_data, input_queue_capacity,
                 num_input_reader_threads, batch_size, dropout, dropout_rate, context_size,
                 mode="train", input_reader=None, input_queue=None):
        """Basic setup. The actual TensorFlow graph is constructed in build().

    Args:
      config: Object containing configuration parameters.
      mode: "train", "eval" or "encode".
      input_reader: Subclass of tf.ReaderBase for reading the input serialized
        tf.Example protocol buffers. Defaults to TFRecordReader.

    Raises:
      ValueError: If mode is invalid.
    """
        if mode not in ["train", "eval", "encode"]:
            raise ValueError("Unrecognized mode: %s" % mode)

        self.config = config
        self.mode = mode
        self.reader = input_reader if input_reader else tf.TFRecordReader()
        self.input_queue = input_queue

        self.input_file_pattern = input_file_pattern
        # Initializer used for non-recurrent weights.-0.1~0.1之间
        self.uniform_initializer = tf.random_uniform_initializer(
            minval=-uniform_init_scale,
            maxval=uniform_init_scale)
        self.shuffle_input_data = shuffle_input_data
        self.input_queue_capacity = input_queue_capacity
        self.num_input_reader_threads = num_input_reader_threads
        self.batch_size = batch_size
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.context_size = context_size

        # Input sentences represented as sequences of word ids. "encode" is the
        # source sentence, "decode_pre" is the previous sentence and "decode_post"
        # is the next sentence.
        # Each is an int64 Tensor with  shape [batch_size, padded_length].
        self.encode_ids = None

        # Boolean masks distinguishing real words (1) from padded words (0).
        # Each is an int32 Tensor with shape [batch_size, padded_length].
        self.encode_mask = None

        # Input sentences represented as sequences of word embeddings.
        # Each is a float32 Tensor with sh ape [batch_size, padded_length, emb_dim].
        self.encode_emb = None

        # The output from the sentence encoder.
        # A float32 Tensor with shape [batch_size, num_gru_units].
        self.thought_vectors = None

        # The total loss to optimize.
        self.total_loss = None

    def build_inputs(self):

        if self.mode == "encode":
            encode_ids = tf.placeholder(tf.int64, (None, None), name="encode_ids")
            encode_mask = tf.placeholder(tf.int8, (None, None), name="encode_mask")
        else:
            # Prefetch serialized tf.Example protos.
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.input_file_pattern,
                shuffle=self.shuffle_input_data,
                capacity=self.input_queue_capacity,
                num_reader_threads=self.num_input_reader_threads)
            print("input_queue", input_queue)
            # Deserialize a batch.
            serialized = input_queue.dequeue_many(self.batch_size)
            print("serialized", serialized)
            encode = input_ops.parse_example_batch(serialized)
            print("encode", encode)

            encode_ids = encode.ids
            encode_mask = encode.mask

        self.encode_ids = encode_ids
        self.encode_mask = encode_mask

    # 这个函数中只看懂了fixed，别的没看懂？？？？？？？？？？？
    def build_word_embeddings(self):
        """word_emb:词向量与词典中元素顺序保持一致
            word_embeddings:随机初始化的词向量数组或者需要扩展的两个词典"""
        rand_init = self.uniform_initializer
        self.word_embeddings = []
        self.encode_emb = []
        self.init = None
        for v in self.config.vocab_configs:
            if v.mode == 'fixed':
                if self.mode == "train":
                    word_emb = tf.get_variable(
                        name=v.name,
                        shape=[v.size, v.dim],
                        trainable=False)
                    embedding_placeholder = tf.placeholder(
                        tf.float32, [v.size, v.dim])
                    embedding_init = word_emb.assign(embedding_placeholder)  # 将word_emb的值用embedding_placeholder代替

                    rand = np.random.rand(1, v.dim)

                    # word_vecs = np.load(v.embs_file)
                    word_vecs = None
                    tf.logging.info("the emb file in %s ", v.embs_file)
                    word_vecs = np.loadtxt(v.embs_file)

                    load_vocab_size = word_vecs.shape[0]  # 词向量的个数
                    print("load_vocab_size: ", load_vocab_size)

                    assert (load_vocab_size == v.size - 1)  # v.size多了一个pad

                    word_init = np.concatenate((rand, word_vecs), axis=0)
                    self.init = (embedding_init, embedding_placeholder, word_init)

                else:
                    word_emb = tf.get_variable(
                        name=v.name,
                        shape=[v.size, v.dim])

                # 词向量必须要和vocab.txt保持一致
                encode_emb = tf.nn.embedding_lookup(word_emb, self.encode_ids)
                print("encode_emb", encode_emb)
                self.word_emb = word_emb
                self.encode_emb.extend([encode_emb, encode_emb])  ##### 两个编码器拼接
                print("self.encode_emb", self.encode_emb)

            if v.mode == 'trained':
                for inout in ["", "_out"]:
                    # 随机初始化一个词向量数组
                    word_emb = tf.get_variable(
                        name=v.name + inout,
                        shape=[v.size, v.dim],
                        initializer=rand_init)
                    if self.mode == 'train':
                        self.word_embeddings.append(word_emb)

                    encode_emb = tf.nn.embedding_lookup(word_emb, self.encode_ids)
                    self.encode_emb.append(encode_emb)

            if v.mode == 'expand':
                for inout in ["", "_out"]:
                    encode_emb = tf.placeholder(tf.float32, (
                        None, None, v.dim), v.name + inout)
                    self.encode_emb.append(encode_emb)
                    word_emb_dict = read_vocab_embs(v.vocab_file + inout + ".txt",
                                                    v.embs_file + inout + ".npy")
                    self.word_embeddings.append(word_emb_dict)  # 两个字典

            if v.mode != 'expand' and self.mode == 'encode':
                word_emb_dict = read_vocab(v.vocab_file)
                self.word_embeddings.extend([word_emb_dict, word_emb_dict])

    def _initialize_cell(self, num_units, cell_type="GRU"):
        if cell_type == "GRU":
            return tf.contrib.rnn.GRUCell(num_units=num_units)
        elif cell_type == "LSTM":
            return tf.contrib.rnn.LSTMCell(num_units=num_units)
        else:
            raise ValueError("Invalid cell type")

    def bow(self, word_embs, mask):
        mask_f = tf.expand_dims(tf.cast(mask, tf.float32), -1)
        word_embs_mask = word_embs * mask_f
        bow = tf.reduce_sum(word_embs_mask, axis=1)  # 求和
        return bow

    def rnn(self, word_embs, mask, scope, encoder_dim, cell_type="GRU"):
        """输出隐层状态值[batch_size,hidden_size]"""
        length = tf.to_int32(tf.reduce_sum(mask, 1), name="length")  # 所有样本的长度列表

        if self.config.bidir:
            if encoder_dim % 2:
                raise ValueError(
                    "encoder_dim must be even when using a bidirectional encoder.")
            num_units = encoder_dim // 2
            print("num_units: ", num_units)
            cell_fw = self._initialize_cell(num_units, cell_type=cell_type)
            cell_bw = self._initialize_cell(num_units, cell_type=cell_type)
            # outputs为(output_fw, output_bw)，是一个包含前向cell输出tensor和
            # 后向cell输出tensor组成的二元组。保存了所有时间步的隐层状态
            # 2表示前向和后向，[2,batch_size, max_time, hidden_size]
            # states为(output_state_fw, output_state_bw)，保存了最后时刻的隐层状态
            # 包含了前向和后向最后的隐藏状态的组成的二元组。
            # 第一个2表示前向和后向,第二个2表示输出的cell和h。[2,2,batch_size,hidden_size]

            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                inputs=word_embs,  # word_embs的维度[batch_size,max_length,word_dim]
                sequence_length=length,
                dtype=tf.float32,
                scope=scope)
            # 如果是LSTM则有两个输出cell和h，如果是GRU则只有一个输出h
            if cell_type == "LSTM":
                states = [states[0][1], states[1][1]]  # 前向隐藏层和后向隐藏层
            state = tf.concat(states, 1)
        else:
            cell = self._initialize_cell(encoder_dim, cell_type=cell_type)
            outputs, state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=word_embs,
                sequence_length=length,
                dtype=tf.float32,
                scope=scope)
            if cell_type == "LSTM":
                state = state[1]
        return state

    def build_encoder(self):
        """Builds the sentence encoder.

    Inputs:
      self.encode_emb
      self.encode_mask

    Outputs:
      self.thought_vectors

    Raises:
      ValueError: if config.bidirectional_encoder is True and config.encoder_dim
        is odd.
    """
        names = ["", "_out"]
        self.thought_vectors = []
        for i in range(2):  # 两个编码器
            with tf.variable_scope("encoder" + names[i]) as scope:

                if self.config.encoder == "gru":
                    sent_rep = self.rnn(self.encode_emb[i], self.encode_mask, scope, self.config.encoder_dim,
                                        cell_type="GRU")
                    print("sent_rep", sent_rep)
                elif self.config.encoder == "lstm":
                    sent_rep = self.rnn(self.encode_emb[i], self.encode_mask, scope, self.config.encoder_dim,
                                        cell_type="LSTM")
                elif self.config.encoder == 'bow':
                    sent_rep = self.bow(self.encode_emb[i], self.encode_mask)
                else:
                    raise ValueError("Invalid encoder")

                thought_vectors = tf.identity(sent_rep, name="thought_vectors")
                self.thought_vectors.append(thought_vectors)

    def build_loss(self):
        """Builds the loss Tensor.
    Outputs:
      self.total_loss
    """
        all_sen_embs = self.thought_vectors
        print("self.thought_vectors: ", self.thought_vectors)

        if self.dropout:
            mask_shp = [1, self.config.encoder_dim]  # [1,1200]
            print("mask_shp", mask_shp)
            # 随机生成判断神经元是否失活
            bin_mask = tf.random_uniform(mask_shp) > self.dropout_rate
            print("bin_mask", bin_mask)
            # if true,返回ones，else 返回zeros
            bin_mask = tf.where(bin_mask, tf.ones(mask_shp), tf.zeros(mask_shp))
            print("bin_mask", bin_mask)
            src = all_sen_embs[0] * bin_mask
            print("src", src)
            dst = all_sen_embs[1] * bin_mask
            print("dst", dst)
            scores = tf.matmul(src, dst, transpose_b=True)
            print("scores", scores)

        else:
            scores = tf.matmul(all_sen_embs[0], all_sen_embs[1], transpose_b=True)  # study pre current post

        # Ignore source sentence，将对角元素全换为0
        scores = tf.matrix_set_diag(scores, np.zeros(self.batch_size))
        print("scores", scores)
        # Targets
        targets_np = np.zeros((self.batch_size, self.batch_size))
        ctxt_sent_pos = list(range(-self.context_size, self.context_size + 1))
        print("ctxt_sent_pos", ctxt_sent_pos)
        ctxt_sent_pos.remove(0)
        print("ctxt_sent_pos", ctxt_sent_pos)
        for ctxt_pos in ctxt_sent_pos:
            targets_np += np.eye(self.batch_size, k=ctxt_pos)
        print("targets_np", targets_np)
        targets_np_sum = np.sum(targets_np, axis=1, keepdims=True)
        print("targets_np_sum", targets_np_sum)
        targets_np = targets_np / targets_np_sum
        print("targets_np", targets_np)
        targets = tf.constant(targets_np, dtype=tf.float32)
        print("targets", targets)

        # Forward and backward scores
        f_scores = scores[:-1]
        print("f_scores", f_scores)
        b_scores = scores[1:]
        print("b_scores", b_scores)

        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=targets, logits=scores)

        loss = tf.reduce_mean(losses)

        # 利用tensorboard可视化，绘图
        tf.summary.scalar("losses/ent_loss", loss)

        self.total_loss = loss

        if self.mode == "eval":
            f_max = tf.to_int64(tf.argmax(f_scores, axis=1))
            b_max = tf.to_int64(tf.argmax(b_scores, axis=1))

            targets = range(self.batch_size - 1)
            targets = tf.constant(list(targets), dtype=tf.int64)
            fwd_targets = targets + 1

            names_to_values, names_to_updates = tf.contrib.slim.metrics.aggregate_metric_map({
                "Acc/Fwd Acc": tf.contrib.slim.metrics.streaming_accuracy(f_max, fwd_targets),  # 上一句
                "Acc/Bwd Acc": tf.contrib.slim.metrics.streaming_accuracy(b_max, targets)  # 下一句准确率
            })

            for name, value in names_to_values.items():
                tf.summary.scalar(name, value)

            self.eval_op = names_to_updates.values()

    def build(self):
        """Creates all ops for training, evaluation or encoding."""
        self.build_inputs()
        self.build_word_embeddings()
        self.build_encoder()  # 两个编码器
        self.build_loss()

    def build_enc(self):
        """Creates all ops for training, evaluation or encoding."""
        self.build_inputs()
        self.build_word_embeddings()
        self.build_encoder()

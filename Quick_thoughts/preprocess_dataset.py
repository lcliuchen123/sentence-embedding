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
"""Converts a set of text files to TFRecord format with Example protos.

Each Example proto in the output contains the following fields:

  decode_pre: list of int64 ids corresponding to the "previous" sentence.
  encode: list of int64 ids corresponding to the "current" sentence.
  decode_post: list of int64 ids corresponding to the "post" sentence.

In addition, the following files are generated:

  vocab.txt: List of "<word> <id>" pairs, where <id> is the integer
             encoding of <word> in the Example protos.
  word_counts.txt: List of "<word> <count>" pairs, where <count> is the number
                   of occurrences of <word> in the input files.

The vocabulary of word ids is constructed from the top --num_words by word
count. All other words get the <unk> word id.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import time
import logging

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_files", "../data/sent_char_n.txt",
                       "Comma-separated list of globs matching the input "
                       "files. The format of the input files is assumed to be "
                       "a list of newline-separated sentences, where each "
                       "sentence is already tokenized.")

tf.flags.DEFINE_string("vocab_file", "vocab.txt",
                       "(Optional) existing vocab file. Otherwise, a new vocab "
                       "file is created and written to the output directory. "
                       "The file format is a list of newline-separated words, "
                       "where the word id is the corresponding 0-based index "
                       "in the file.")

tf.flags.DEFINE_string("output_dir", "../output/transformer/", "Output directory.")

tf.flags.DEFINE_integer("train_output_shards", 10,
                        "Number of output shards for the training set.")

tf.flags.DEFINE_integer("validation_output_shards", 1,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_integer("num_validation_sentences", 50000,
                        "Number of output shards for the validation set.")

tf.flags.DEFINE_float("split_rate", 0.2, "the rate of split the train and validation")

# ???????训练集的句子数量
tf.flags.DEFINE_integer("num_train_inst", 1000000, "Number of training instances")

# num_words是只保存频率最高的一部分字或者词语（从训练集里面选取的）
tf.flags.DEFINE_integer("num_words", 20000,
                        "Number of words to include in the output.")

# 主要是用来控制最大句子数量
tf.flags.DEFINE_integer("max_sentences", 0,
                        "If > 0, the maximum number of sentences to output.")

tf.flags.DEFINE_integer("max_sentence_length", 30,
                        "If > 0, exclude sentences whose encode, decode_pre OR"
                        "decode_post sentence exceeds this length.")

tf.flags.DEFINE_boolean("case_sensitive", False,
                        "Use case sensitive vocabulary")

tf.logging.set_verbosity(tf.logging.INFO)

UNK = 'UNNK'
UNK_ID = 1


# 自己给定词典和预先训练好的词向量或者利用训练集生成词典
def _build_vocabulary(input_files, max_num):
    """Loads or builds the model vocabulary.
  Args:
    input_files: List of pre-tokenized input .txt files.

  Returns:
    vocab: A dictionary of word to id.
  """
    vocab_dir = os.path.join("../data/", input_files[0].split('/')[-1][:-4])

    if FLAGS.vocab_file:
        vocab_file = os.path.join(vocab_dir, FLAGS.vocab_file)
        if os.path.exists(vocab_file):
            tf.logging.info("Loading existing vocab file: %s !", vocab_file)
        vocab = collections.OrderedDict()
        vocab['<pad>'] = 0
        with tf.gfile.GFile(vocab_file, mode="r") as f:
            for i, line in enumerate(f):
                word = line.encode('utf-8').decode("utf-8").strip()
                if word in vocab:
                    print('Duplicate word:', word)
                # assert word not in vocab, "Attempting to add word twice: %s" % word,UNNK的索引为1，把索引0留给padding
                vocab[word] = i + 1
        tf.logging.info("Read vocab of size %d from %s",
                        len(vocab), FLAGS.vocab_file)
        return vocab

    tf.logging.info("Creating vocabulary.")

    num = 0
    output_dir = ""  # 输出目录
    sentence_seq_length = []
    wordcount = collections.Counter()
    for input_file in input_files:
        tf.logging.info("Processing file: %s", input_file)
        # input_file: ../data/sent_word_n.txt
        output_dir = os.path.join(FLAGS.output_dir, input_file.split('/')[-1][:-4])
        if not tf.gfile.IsDirectory(output_dir):
            tf.gfile.MakeDirs(output_dir)
        tf.logging.info("the output_dit: %s have been created! ", output_dir)

        for sentence in tf.gfile.FastGFile(input_file):
            sentence = sentence.strip()
            sentence = [word.strip() for word in sentence.split()]
            total_length = sum(len(word) for word in sentence)
            # 如果长度超过1000，直接换下一个
            if total_length > 1000:
                continue

            if len(sentence) > 1:
                # max_num限制选取句子的最大数目
                if num >= max_num:
                    tf.logging.info("the %d sentence have been processed! And the data have been done", max_num)
                    break

                wordcount.update(sentence)
                sentence_seq_length.append(len(sentence))

                num += 1
                if num % 100000 == 0:
                    tf.logging.info("Processed %d sentences", num)

    # 统计每条句子包含词的数量
    sentence_seq_length = sorted(sentence_seq_length)
    sen_length_list = np.array(sentence_seq_length)
    sen_file_name = os.path.join(output_dir, "sen_length.txt")
    np.savetxt(sen_file_name, sen_length_list, fmt='%d')

    tf.logging.info("the min length of sentences are %d", min(sentence_seq_length))  #
    tf.logging.info("the max length of sentences are %d", max(sentence_seq_length))  # 779
    tf.logging.info("Processed %d sentences total", num)  # 9871094

    words = list(wordcount.keys())
    freqs = list(wordcount.values())
    sorted_indices = np.argsort(freqs)[::-1]

    vocab = collections.OrderedDict()
    vocab[UNK] = UNK_ID
    for w_id, w_index in enumerate(sorted_indices[0:FLAGS.num_words - 2]):
        vocab[words[w_index]] = w_id + 1  # 0: <unk>

    tf.logging.info("Created vocab with %d words", len(vocab))

    vocab_file = os.path.join(output_dir, "vocab.txt")
    with tf.gfile.FastGFile(vocab_file, "w") as f:
        f.write("\n".join(vocab.keys()))
    tf.logging.info("Wrote vocab file to %s", vocab_file)

    word_counts_file = os.path.join(output_dir, "word_counts.txt")
    with tf.gfile.FastGFile(word_counts_file, "w") as f:
        for i in sorted_indices:
            f.write("%s %d\n" % (words[i], freqs[i]))
    tf.logging.info("Wrote word counts file to %s", word_counts_file)

    return vocab


def _int64_feature(value):
    """Helper for creating an Int64 Feature."""
    return tf.train.Feature(int64_list=tf.train.Int64List(
        value=[int(v) for v in value]))


def _sentence_to_ids(sentence, vocab):
    """Helper for converting a sentence (list of words) to a list of ids."""
    if FLAGS.case_sensitive:
        ids = [vocab.get(w, UNK_ID) for w in sentence]
    else:
        ids = [vocab.get(w.lower(), UNK_ID) for w in sentence]
    return ids


# def _create_serialized_example(predecessor, current, successor, vocab):
def _create_serialized_example(current, vocab):
    """Helper for creating a serialized Example proto."""
    # example = tf.train.Example(features=tf.train.Features(feature={
    #    "decode_pre": _int64_feature(_sentence_to_ids(predecessor, vocab)),
    #    "encode": _int64_feature(_sentence_to_ids(current, vocab)),
    #    "decode_post": _int64_feature(_sentence_to_ids(successor, vocab)),
    # }))
    example = tf.train.Example(features=tf.train.Features(feature={
        "features": _int64_feature(_sentence_to_ids(current, vocab)),
    }))
    # example = tf.train.Example(features=tf.train.Features(feature=
    #    _int64_feature(_sentence_to_ids(current, vocab)),
    # ))

    return example.SerializeToString()  # 序列化保存数据，便于后期操作


def _process_input_file(filename, vocab, stats):
    """Processes the sentences in an input file.

  Args:
    filename: Path to a pre-tokenized input .txt file.
    vocab: A dictionary of word to id.

  Returns:
    processed: A list of serialized Example protos
  """

    tf.logging.info("Processing input file: %s", filename)
    processed = []
    seg_length_list = []
    i = 0
    count = 0
    for sentence_str in tf.gfile.FastGFile(filename):
        sentence_tokens = sentence_str.split()
        total_length = sum([len(word.strip()) for word in sentence_tokens])
        if total_length > 1000:  # 如果句子长度超过1000，删除
            continue

        if len(sentence_tokens) > 1:
            if i >= FLAGS.num_train_inst:
                break

            seg_length_list.append(len(sentence_tokens))

            # 长度超过最大长度的部分会被截断，如果小于最大长度，保持原来长度不变
            sentence_tokens = sentence_tokens[:FLAGS.max_sentence_length]

            less_length = len(sentence_tokens)
            if less_length < 30:
                count += 1

            i += 1
            serialized = _create_serialized_example(sentence_tokens, vocab)
            processed.append(serialized)

            stats.update(["sentence_count"])  # sentence_count自己跟随for循环计数
    tf.logging.info("***********the %d sentences is less than 30 ************", count)
    tf.logging.info("the min length of sentences is %d, the max length of "
                    "sentences is %d !", min(seg_length_list), max(seg_length_list))
    tf.logging.info("stats['sentence_count']: %d", stats["sentence_count"])

    tf.logging.info("Completed processing file %s", filename)
    return processed


def _write_shard(filename, dataset, indices):
    """Writes a TFRecord shard."""
    with tf.python_io.TFRecordWriter(filename) as writer:
        for j in indices:
            writer.write(dataset[j])


def _write_dataset(name, dataset, indices, num_shards, input_file):
    """Writes a sharded TFRecord dataset.

  Args:
    name: Name of the dataset (e.g. "train").
    dataset: List of serialized Example protos.
    indices: List of indices of 'dataset' to be written.
    num_shards: The number of output shards.即生成几个TFRECORD文件
  """
    output_dir = os.path.join(FLAGS.output_dir, input_file.split('/')[-1][:-4])
    if not tf.gfile.IsDirectory(output_dir):
        tf.gfile.MakeDirs(output_dir)
    tf.logging.info("the output_dit: %s have been created! ", output_dir)

    tf.logging.info("Writing dataset %s", name)
    borders = np.int32(np.linspace(0, len(indices), num_shards + 1))  # 创建num_shards+1的等差数列
    for i in range(num_shards):
        filename = os.path.join(output_dir, "%s-%.5d-of-%.5d" % (name, i,
                                                                 num_shards))
        shard_indices = indices[borders[i]:borders[i + 1]]
        _write_shard(filename, dataset, shard_indices)
        tf.logging.info("Wrote dataset indices [%d, %d) to output shard %s",
                        borders[i], borders[i + 1], filename)
    tf.logging.info("Finished writing %d sentences in dataset %s.",
                    len(indices), name)


def main(unused_argv):
    if not FLAGS.input_files:
        raise ValueError("--input_files is required.")
    if not FLAGS.output_dir:
        raise ValueError("--output_dir is required.")

    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    input_files = []
    for pattern in FLAGS.input_files.split(","):
        tf.logging.info("the pattern is %s", pattern)
        match = tf.gfile.Glob(pattern)  # 返回一个list
        if not match:
            raise ValueError("Found no files matching %s" % pattern)
        # input_files.extend(match)
        input_files.append(match)

    tf.logging.info("Found %d input files.", len(input_files))
    print("input_files: ", input_files)

    for input_file in input_files:
        tf.logging.info("**************************************")
        tf.logging.info("starting process the %s !!!!!!!", input_file)
        start_time = time.time()
        # 如果不存在词典文件，最多返回前一百万个样本中的词作为词典
        vocab = _build_vocabulary(input_file, 1000000)
        tf.logging.info("Generating dataset.")
        stats = collections.Counter()
        dataset = []
        for filename in input_file:
            dataset.extend(_process_input_file(filename, vocab, stats))
            if FLAGS.max_sentences and stats["sentence_count"] >= FLAGS.max_sentences:
                break

        tf.logging.info("Generated dataset with %d sentences.", len(dataset))
        for k, v in stats.items():
            tf.logging.info("%s: %d", k, v)

        # tf.logging.info("Shuffling dataset.")
        # np.random.seed(123)
        # shuffled_indices = np.random.permutation(len(dataset))
        # val_indices = shuffled_indices[:FLAGS.num_validation_sentences]
        # train_indices = shuffled_indices[FLAGS.num_validation_sentences:]

        indices = range(len(dataset))
        # val_indices = indices[:FLAGS.num_validation_sentences]
        # train_indices = indices[FLAGS.num_validation_sentences:]
        val_indices = indices[:int(FLAGS.split_rate * len(dataset))]
        train_indices = indices[int(FLAGS.split_rate * len(dataset)):]

        _write_dataset("train", dataset, train_indices, FLAGS.train_output_shards, input_file[0])
        _write_dataset("validation", dataset, val_indices,
                       FLAGS.validation_output_shards, input_file[0])

        end_time = time.time()
        cost_time = end_time - start_time
        tf.logging.info("the file have been created! And "
                        "the cost time of %s is %f. ", input_file, cost_time)


if __name__ == "__main__":
    tf.app.run()

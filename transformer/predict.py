
import sys
sys.path.append('.')
sys.path.append("..")
import collections
import tensorflow as tf
import numpy as np
from transformer.model import s2v_model
from run_time import *
unk = 'UNNK'
tf.logging.set_verbosity(tf.logging.INFO)


# 1.获取字典
def get_vocab(vocab_file):
    # 按照插入的顺序对字典进行排序
    vocab = collections.OrderedDict()
    with tf.gfile.GFile(vocab_file, mode="r") as f:
        for i, line in enumerate(f):
            word = line.encode('utf-8').decode("utf-8").strip()
            if word in vocab:
                tf.logging.info('Duplicate word: %s', word)
            # assert word not in vocab, "Attempting to add word twice: %s" % word
            vocab[word] = i + 1
    tf.logging.info("Read vocab of size %d from %s",
                    len(vocab), vocab_file)
    return vocab


# 2.将待预测的文本（用空格隔开）转换成对应的索引
def get_word_index(sentences, vocab):
    """
    sentences: 一个句子列表，每个句子进行过预处理，按照空格隔开
    vocab: 一个字典，key是词，value为对应的索引
    return：一个句子索引列表
    """
    sent_index = []
    if sentences:
        for sent in sentences:
            seq_index = [vocab.get(word.strip(), vocab[unk]) for word in sent.split()]
            sent_index.append(seq_index)

    return sent_index


# 3. 进行padding和mask
def batch_padding(sent_index):
    length_list = []
    for index in sent_index:
        length_list.append(len(index))
    max_seq_length = max(length_list)
    batch_size = len(sent_index)
    pad_sentences = np.zeros((batch_size, max_seq_length))
    mask = np.zeros((batch_size, max_seq_length))
    for i in range(batch_size):
        length = len(sent_index[i])
        if length > max_seq_length:
            pad_sentences[i][:] = sent_index[i][:max_seq_length]
            mask[i][:max_seq_length] = 1
        else:
            pad_sentences[i][:length] = sent_index[i]
            mask[i][:length] = 1
    return pad_sentences, mask


# 4.加载模型,restore
def load_model(checkpoint_path, saver):
    if tf.gfile.IsDirectory(checkpoint_path):
        latest_checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        checkpoint_path = latest_checkpoint_path
        if not latest_checkpoint_path:
            raise ValueError("the latest checkpoint path not found!!!")

    def restore(sess):
        saver.restore(sess, checkpoint_path)

    return restore


def build_model(data, checkpoint_path, mode, batch_size, vocab_size,
                input_file_pattern, emb_file, num_head, num_units_list, uniform_init=0.1):
    s2v_list = []
    # 构建模型
    g = tf.Graph()
    with g.as_default():
        tf.logging.info("*******Building the model****************")
        model = s2v_model(mode, batch_size, input_file_pattern, uniform_init, emb_file, num_head, num_units_list,
                          num_input_reader_threads=1, input_queue_capacity=640000, vocab_size=vocab_size)
        output = model.build_encode()
        saver = tf.train.Saver()
        tf.logging.info("***********loading the latest model!***************")
        restore_model = load_model(checkpoint_path, saver)

        # sess = tf.Session(graph=g), 对句子进行编码，生成句向量
        with tf.Session(graph=g) as sess:
            restore_model(sess)

            batch_indices = np.arange(0, len(data), batch_size)
            for index, batch_start_index in enumerate(batch_indices):
                if index % 100 == 0:
                    tf.logging.info("********the %d batch, and the %d index **********", index, batch_start_index)

                input_ids, input_masks = batch_padding(data[batch_start_index:batch_start_index+batch_size])
                print("the shape of input_ids is: ", input_ids.shape)
                print("the shape of input_masks is: ", input_masks.shape)

                feed_dict = {"encode_ids:0": input_ids, "encode_mask:0": input_masks}
                s2v = sess.run(output, feed_dict=feed_dict)
                # 输出的结果维度为：[batch_size, seq_length, dim],
                # 本文对其进行简单平均或直接相加或者标准化后再求和、平均
                # 标准化后平均，对每个batch进行标准化
                # s2v = np.reshape(s2v, (-1, 300))
                # s2v = (s2v - np.mean(s2v, axis=0))/(np.var(s2v, axis=0)**0.5)
                # s2v = np.reshape(s2v, (-1, input_ids.shape[-1], 300))
                s2v = np.sum(s2v, axis=1) / input_ids.shape[-1]
                # s2v = np.sum(s2v, axis=1)
                s2v_list.extend(s2v)

    # 简单平均后标准化
    # print("the mean shape of s2v_list is: ", np.mean(s2v_list, axis=0).shape)
    # s2v_list = (s2v_list - np.mean(s2v_list, axis=0)) / (np.var(s2v_list, axis=0) ** 0.5)

    # 求和后标准化
    # s2v_list = (s2v_list - np.mean(s2v_list, axis=0)) / (np.var(s2v_list, axis=0) ** 0.5)

    print("the shape of s2v_list is: ", len(s2v_list), s2v_list[0].shape)

    return s2v_list


@cost_time
def get_s2v(sentences, vocab_file, checkpoint_path, emb_file, num_head, num_units_list,
            vocab_size, input_file_pattern=None, batch_size=128, mode='encode'):
    """输出的结果维度为：[batch_size, seq_length, dim],
    本文对其进行简单平均或者利用最后一个词的词向量"""
    vocab = get_vocab(vocab_file)
    sent_index = get_word_index(sentences, vocab)

    # 字典中没有包含pad字符对应的索引
    assert len(vocab) + 1 == vocab_size

    result = build_model(sent_index, checkpoint_path, mode, batch_size, vocab_size,
                         input_file_pattern, emb_file, num_head, num_units_list)

    return result


if __name__ == "__main__":
    sentences = ["国家 卫生 健康 委员会 召开 新闻 发布会 "
                                  "介绍 各地 医疗队 支援 湖北 抗击 新型 冠状 病毒 感染 肺炎"]
    # vocab_file = "vocab.txt"
    # vocab = get_vocab(vocab_file)
    # sent_index = get_word_index(sentences, vocab)
    # input_ids, input_mask = batch_padding(sent_index, 30)
    # print(input_ids)
    # print(input_mask)
    vocab_file = "../data/new_sent_word_rem/vocab.txt"
    vocab_size = 20001
    checkpoint_path = "../model/train/second_train/new_sent_word_rem/transformer/"
    emb_file = None
    # emb_file = "../data/sent_word_rem/emb.txt"
    # vocab = get_vocab(vocab_file)
    # sent_index = get_word_index(sentences, vocab)
    # input_ids, input_mask = batch_padding(sent_index)
    # print(input_ids)
    # print(input_mask)
    num_head = 2
    num_units_list = [1200, 300]
    result = get_s2v(sentences, vocab_file, checkpoint_path, emb_file, num_head, num_units_list, vocab_size)
    print("result: ", result)
    print("the dim of result is：", result[0].shape)

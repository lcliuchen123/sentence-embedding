
from __future__ import print_function
import numpy as np


def getWordmap(textfile):
    """textfile是词向量文件列表，可能一个是.vector,另一个是.npy文件
    当词向量文件过大时用np.load加载.npy文件
    每一行的第一列是词，后面是词向量，以空格隔开
    返回一个词典（所有词）和词向量列表"""
    words = {}
    embeding_size = 0
    lines = []
    i = 0
    We = []
    word_file = textfile[0]
    with open(word_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.split()
            if len(line) > 2:
                embeding_size = len(line) - 1
                words[line[0]] = i
                if len(textfile) == 1:
                    v = [float(num) for num in line[1:]]
                    We.append(v)
                i += 1
    print("the embedding size is %d ! " % embeding_size)
    words['UNK'] = len(lines)  # 未出现在字典里的词
    if len(textfile) == 2:
        npy_file = textfile[1]
        We = np.load(npy_file)  # 如果文件过大，无法创建数组
        assert We.shape[0] == i
        assert We.shape[1] == embeding_size
    unk = np.zeros((1, embeding_size))
    We = np.concatenate((We, unk), axis=0)

    return words, We


def prepare_data(list_of_seqs):
    """
    :param list_of_seqs: 句子列表,索引
    :return: x[i,:]：表示第i个句子的索引列表
    """
    lengths = [len(s) for s in list_of_seqs]
    n_samples = len(list_of_seqs)
    maxlen = np.max(lengths)
    x = np.zeros((n_samples, maxlen)).astype('int32')
    x_mask = np.zeros((n_samples, maxlen)).astype('float32')
    for idx, s in enumerate(list_of_seqs):
        x[idx, :lengths[idx]] = s
        x_mask[idx, :lengths[idx]] = 1.
    x_mask = np.asarray(x_mask, dtype='float32')
    return x, x_mask


def lookupIDX(words,w):
    """
    如果词没有出现在词典中，记为UNK,词向量为0
    返回词w在词典words中的索引"""
    w = w.lower()
    if len(w) > 1 and w[0] == '#':
        w = w.replace("#", "")
    if w in words:
        return words[w]
    else:
        with open("./oov.txt", 'w', encoding='utf-8') as f:
            f.write(w + '\n')
        return words['UNK']


def getSeq(p1, words):
    """
    :param p1: 句子
    :param words: 词典
    :return: p1在词典中的索引列表
    """
    p1 = p1.split()
    X1 = []
    for i in p1:
        X1.append(lookupIDX(words, i))
    return X1


def sentences2idx(sentences, words):
    """
    Given a list of sentences, output array of word indices that can be fed into the algorithms.
    :param sentences: a list of sentences
    :param words: a dictionary, words['str'] is the indices of the word 'str'
    :return: x1, m1. x1[i, :] is the word indices in sentence i, m1[i,:] is the mask for sentence i
    (0 means no word at the location)
    """
    seq1 = []
    for i in sentences:
        seq1.append(getSeq(i, words))
    x1, m1 = prepare_data(seq1)
    return x1, m1


def getWordWeight(weightfile, a=1e-3):
    """
    :param weightfile: each line is a word and its frequency
    :param a: 权重中的常数
    :return: 每个词对应的权重
    """
    if a <= 0:  # when the parameter makes no sense, use unweighted
        a = 1.0

    word2weight = {}
    with open(weightfile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    N = 0
    for i in lines:
        i = i.strip()
        if len(i) > 0:
            i = i.split()
            if len(i) == 2:
                word2weight[i[0]] = float(i[1])
                N += float(i[1])
            else:
                print("no freque word: ", i)
    for key, value in word2weight.items():
        word2weight[key] = a / (a + value/N)
    return word2weight


def getWeight(words, word2weight):
    """返回字典中每个词的权重，weight4ind中第i个值表示词典中第i个词的权重
    words:词典
    word2weight:字典:key: 词 value：权重"""
    weight4ind = {}
    for word, ind in words.items():
        if word in word2weight:
            weight4ind[ind] = word2weight[word]
        else:
            weight4ind[ind] = 1.0
    return weight4ind


def seq2weight(seq, mask, weight4ind):
    """
    :param seq: x[i, :] are the indices of the words in sentence i，所有句子中词对应的索引n*m
    :param mask: 标记那个位置是否存在单词
    :param weight4ind: weight4ind中第i个值表示词典中第i个词的权重
    :return:  权重矩阵
    """
    weight = np.zeros(seq.shape).astype('float32')
    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            if mask[i, j] > 0 and seq[i, j] >= 0:
                weight[i, j] = weight4ind[seq[i, j]]
    weight = np.asarray(weight, dtype='float32')
    return weight

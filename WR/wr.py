
import sys
sys.path.append('.')
sys.path.append('..')
from sklearn.decomposition import TruncatedSVD
from WR.wr_pre import *
from run_time import *
from gensim.models import Word2Vec
import numpy as np


@cost_time
def get_weighted_average(We, x, w):
    """
    创建一个包含所有词向量的列表，利用每个句子的索引，表示每个句子每个位置的词的权重，
    还引入mask标记该位置是否有单词。
    Compute the weighted average vectors
    300是词向量维度，n是句子数，m是所有词的个数
    :param We: We[i,:] is the vector for word i ,所有词对应的词向量m*300
    :param x: x[i, :] are the indices of the words in sentence i，所有句子中词对应的索引n*m
    :param w: w[i, :] are the weights for the words in sentence i.所有单词的权重n*m
    :return: emb[i, :] are the weighted average vector for sentence i
    """
    n_samples = x.shape[0]
    print(We.shape)
    emb = np.zeros((n_samples, We.shape[1]))
    for i in range(n_samples):
        emb[i,:] = w[i,:].dot(We[x[i,:],:]) / np.count_nonzero(w[i,:])
    return emb


def compute_pc(X,npc=1):
    """
    Compute the principal components. DO NOT MAKE THE DATA ZERO MEAN!
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: component_[i,:] is the i-th pc
    """
    svd = TruncatedSVD(n_components=npc, n_iter=7, random_state=0)
    svd.fit(X)
    return svd.components_


def remove_pc(X, npc=1):
    """
    Remove the projection on the principal components
    :param X: X[i,:] is a data point
    :param npc: number of principal components to remove
    :return: XX[i, :] is the data point after removing its projection
    """
    # dot是矩阵乘积，*是直接相乘,(n,1)*(1,300) = (n,300)
    # 1个主成分（1，300),两个主成分（2，300）
    pc = compute_pc(X, npc)
    if npc==1:
        XX = X - X.dot(pc.transpose()) * pc
    else:
        XX = X - X.dot(pc.transpose()).dot(pc)
    return XX


@cost_time
def SIF_embedding(We, x, w, params):
    """
    Compute the scores between pairs of sentences using weighted average + removing the projection on the first principal component
    :param We: We[i,:] is the vector for word i
    :param x: x[i, :] are the indices of the words in the i-th sentence
    :param w: w[i, :] are the weights for the words in the i-th sentence
    :param params: if >0, remove the projections of the sentence embeddings to their first principal component
    :return: emb, emb[i, :] is the embedding for sentence i
    """
    emb = get_weighted_average(We, x, w)
    if params > 0:
        emb = remove_pc(emb, params)
    return emb


@cost_time
def get_sen2vec(sentences, word2vec_file, word_freque_file, weight_para=1e-3):
    """
    :param sentences:句子列表，每个句子中每个词用空格隔开
    :param word2vec_file: 词向量文件的每一行，第一列是词，后面是词向量，以空格隔开
    :param word_freque_file: each line is a word and its frequency
    :param weight_para: a
    :return:
    """
    # load word vectors
    words, We = getWordmap(word2vec_file)
    # load word weights
    word2weight = getWordWeight(word_freque_file, weight_para) # word2weight['str'] is the weight for the word 'str'
    weight4ind = getWeight(words, word2weight)  # weight4ind[i] is the weight for the i-th word
    # load sentences
    x, m = sentences2idx(sentences, words)
    # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
    w = seq2weight(x, m, weight4ind) # get word weights
    sent_vector = SIF_embedding(We, x, w, 1)
    # print(sent_vector)
    print("the weighted average of word embedding is been created ！")

    return sent_vector


@cost_time
def get_simple_average(sentences, model_name, embedding_size=300):
    """基于词向量的简单平均算法"""
    n_samples = len(sentences)
    sen2vec = []
    model = Word2Vec.load(model_name)
    for i in range(n_samples):
        if sentences[i]:
            sentence = sentences[i].strip()
            v = np.zeros(embedding_size)
            i = 0
            for word in sentence:
                word = word.strip()
                if word:
                    try:
                        vec = model[word]
                        v += vec
                        i += 1
                    except Exception as e:
                        with open("oov.txt", 'w', encoding= 'utf-8') as f:
                            f.write(str(e) + '\n')
                        continue
            if i == 0:
                v = [0] * embedding_size
            else:
                v = v/i
            sen2vec.append(v)
    print("the simple average of word embedding is been created !")

    return sen2vec


if __name__ == "__main__":
    word2vec_file = '../model/word2vec/sent_char_rem.txt_model_300.vector'
    word_freque_file = "../data/sent_char_rem/word_frequent.txt"
    sentences = ["十 秒 之 内 把 所 有 垃 圾 清 理 玩", "可 以 把 垃 圾 清 理 掉 吧"]
    # word_freque = get_word_frequency('./data/char.txt')
    # t = threading.Thread(target=get_word_frequency,args= ("./data/char.txt",))
    # t.start()
    sen2vec = get_sen2vec(sentences, word2vec_file, word_freque_file)
    # t = threading.Thread(target=get_sen2vec, args=(sentences,word2vec_file,word_freque_file))
    # t.start()
    print(sen2vec.shape)

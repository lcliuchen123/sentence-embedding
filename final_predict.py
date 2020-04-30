import sys

sys.path.append('.')
sys.path.append('..')
from WR.wr import *
from Quick_thoughts.predict import *
from sklearn import metrics
from process.get_predata import *
from run_time import *
import pandas as pd
import json
from transformer.predict import *


# 第一步：对真实数据进行预处理得到候选数据列表和预定义数据列表，得到用空格隔开的字或者词
# 停词表去除的单词较多
@cost_time
def preprocess_file(file_name, init_file, cut=True, remove_flag=True):
    """返回候选数据的列表和预定义数据的列表"""
    vocab = get_pre_vocab('./data/label.txt')
    # print("vocab: ", vocab)

    # 待预测的候选数据集
    cand_sent, cand_label = get_pre_data(file_name, vocab, cut=cut, remove_flag=remove_flag,
                                         stop_file_name='./data/stoplist.txt')

    # 预定义操作的数据集
    init_sent, init_label = get_pre_data(init_file, vocab, cut=cut, remove_flag=remove_flag,
                                         stop_file_name='./data/stoplist.txt')

    assert len(cand_sent) == len(cand_label)
    assert len(init_sent) == len(init_label)

    print("the cand_sent length is %d, and the init_sent length is %d ! "
          % (len(cand_sent), len(init_sent)))

    # print("cand_label: ", cand_label)
    # print("init_label: ", init_label)

    return cand_sent, cand_label, init_sent, init_label


# 第二步：调用WR和Quick-thoughts生成句向量
@cost_time
def get_sent_vector(sentences, pattern=None, cut=True, remove_flag=True, paremeter_file='./paramers.json'):
    sen2vec = None

    with open(paremeter_file) as config_file:
        config = json.load(config_file)

    if pattern == 'sa':
        if cut:
            # 1.基于词向量的简单平均
            sen2vec = get_simple_average(sentences, config["word_model_name"])
        else:
            # 2.基于字向量的简单平均
            sen2vec = get_simple_average(sentences, config["char_model_name"])
    elif pattern == 'wr':
        if cut:
            # 3.基于词向量的加权平均
            sen2vec = get_sen2vec(sentences, config["word_word2vec_file"], config["word_word_freque_file"])
        else:
            # 4.基于字向量的加权平均
            sen2vec = get_sen2vec(sentences, config["char_word2vec_file"], config["char_word_freque_file"])
    elif pattern == 'cw_wr':
        # 5.基于字向量和词向量的加权平均
        wr_char_sen2vec = get_sen2vec(sentences, config["char_word2vec_file"], config["char_word_freque_file"])
        wr_word_sen2vec = get_sen2vec(sentences, config["word_word2vec_file"], config["word_word_freque_file"])
        sen2vec = np.concatenate((wr_char_sen2vec, wr_word_sen2vec), axis=1)

    elif pattern == 'qt':
        if cut:
            word2vec_path = config["word2vec_path"]
            model_config = config["model_config"]
            input_file_pattern = config["input_file_pattern"]

        else:
            word2vec_path = config["word2vec_path"].replace("word", "char")
            model_config = config["model_config"].replace("word", "char")
            input_file_pattern = config["input_file_pattern"].replace("word", "char")

        print("the word2vec_path is %s, the mode_config is %s, and the input_file_pattern is %s"
              % (word2vec_path, model_config, input_file_pattern))
        sen2vec = get_result(sentences, word2vec_path, model_config, input_file_pattern)
        print("the length of sen2vec: ", len(sen2vec))
        print("the shape of sen2vec: ", sen2vec[0].shape)

    elif pattern == 'tr':
        num_head = 2
        num_units_list = [1200, 300]
        if cut:
            vocab_file = config["vocab_file"]
            checkpoint_path = config["checkpoint_path"]
            emb_file = config["emb_file"]
            vocab_size = config["word_vocab_size"]
        else:
            vocab_file = config["vocab_file"].replace("word", 'char')
            checkpoint_path = config["checkpoint_path"].replace("word", 'char')
            emb_file = config["emb_file"]
            if emb_file:
                emb_file = config["emb_file"].replace("word", 'char')

            vocab_size = config["char_vocab_size"]

        sen2vec = get_s2v(sentences, vocab_file, checkpoint_path, emb_file, num_head, num_units_list, vocab_size)

    # 句向量列表里面是numpy数组
    if isinstance(sen2vec, list):
        print("the length of sen2vec: ", len(sen2vec))
        print("the shape of sen2vec: ", sen2vec[0].shape)
    else:
        print("the shape of sen2vec: ", sen2vec.shape)

    return sen2vec


# 第三步：计算与预定义操作的余弦相似度，选择相似度最高的作为最终预测结果。
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
    if denom == 0:
        cos = 0
    else:
        cos = num / denom
    sim = 0.5 + 0.5 * cos  # 归一化.cos-(-1)/(1-(-1))=(cos+1)/2=0.5cos+0.5
    return sim


@cost_time
def get_sim(cand_vec, init_vec, thresold=0.5):
    """
    返回最相似的类别对应的标签
    :param cand_vec: 标注样本句向量
    :param init_vec: 预先定义的句向量
    thresold：如果相似度低于该值，则被认为没有对应的预定义操作
    """
    result = []
    max_sim_list = []

    for cand in cand_vec:
        sim_list = []
        for init in init_vec:
            sim = cos_sim(cand, init)
            sim_list.append(sim)
        # 判断是否存在相同的最大值
        t = []
        max_sim = max(sim_list)
        max_sim_list.append(max_sim)
        # 如果低于该阈值则设为其它，即没有对应的预定义操作，利用-1表示
        if max_sim < thresold:
            t = [(-1, max_sim)]
        else:
            if sim_list.count(max_sim) > 1:
                t = [(index, s) for index, s in enumerate(sim_list) if s == max_sim]
                # print("more than one max_sim: ", t)
            else:
                t = [(sim_list.index(max_sim), max_sim)]
        result.append(t)
    print("the max sim in max_sim_list is %f, and the min sim in max_sim_list is %f! "
          % (max(max_sim_list), min(max_sim_list)))
    df = pd.DataFrame(max_sim_list)
    print("df: ", df.describe())
    return result, max_sim_list


# 第四步：计算F1值，统计运行时间。
@cost_time
def get_f1(result, init_label, real_index):
    """
    关心的时有多少真实的预定义操作语句被识别出来，有多少真实无预定义操作的语句被识别出来
    前1500条样本是正例，后500条是负例
    result:余弦相似度结果列表，可能包含多个最大值对应的相似度及标签索引
    init_label: 预定义的标签索引列表，判断句子最相似的预定义语句
    real_index: 待预测数据的真实标签
    """
    # 按照真实标签转化为两类：有预定义操作和无预定义操作，real只包含1和0
    real = [1] * len(real_index)
    print("the max index of cand data is %d ! " % max(real_index))
    for j in range(len(real_index)):
        if real_index[j] == max(real_index):
            real[j] = 0
    # print("real: ", real)

    # pre_list时预测标签，只包含1和0
    pre_list = [0] * len(result)
    for i in range(len(result)):
        for item in result[i]:
            # 负例：无预定义操作，即其它，当预测标签索引为其它时，如果与真实标签相等记为0，反之则记为1.
            if real[i] == 0 and item[0] != -1:
                pre_list[i] = 1
                break
            # 正例：有预定义操作.当预测标签索引不为其它，且与真实标签不等时，记为0
            # 严格意义上真实标签不为其它时，就算不等也应记为1
            if real[i] == 1:
                if item[0] != -1 and init_label[item[0]] == real_index[i]:
                    pre_list[i] = 1
                    break
    # print("pre_list: ", pre_list)

    acc = metrics.precision_score(real, pre_list)
    recall = metrics.recall_score(real, pre_list)
    f1 = metrics.f1_score(real, pre_list)
    print("the accury is %f, the recall is %f, the f1 is %f!" % (acc, recall, f1))

    return f1


@cost_time
def get_one_result(file_name, init_file, pattern, cut=True, remove_flag=True):
    # 1.预处理
    cand_sent, cand_label, init_sent, init_label = preprocess_file(file_name, init_file,
                                                                   cut, remove_flag)
    # 2.生成句向量
    if pattern == 'qt_tr':
        cand_qt_s2v = get_sent_vector(cand_sent, pattern='qt',
                                      cut=cut, remove_flag=remove_flag)
        cand_tr_s2v = get_sent_vector(cand_sent, pattern='tr',
                                      cut=cut, remove_flag=remove_flag)
        cand_qt_tran_sen2vec = np.concatenate((cand_qt_s2v, cand_tr_s2v), axis=1)

        init_qt_s2v = get_sent_vector(init_sent, pattern='qt',
                                      cut=cut, remove_flag=remove_flag)
        init_tr_s2v = get_sent_vector(init_sent, pattern='tr',
                                      cut=cut, remove_flag=remove_flag)
        init_qt_tran_sen2vec = np.concatenate((init_qt_s2v, init_tr_s2v), axis=1)
    else:
        cand_qt_tran_sen2vec = get_sent_vector(cand_sent, pattern=pattern,
                                               cut=cut, remove_flag=remove_flag)
        init_qt_tran_sen2vec = get_sent_vector(init_sent, pattern=pattern, cut=cut,
                                               remove_flag=remove_flag)

    # 3.计算余弦相似度
    # 主要是为了获取最小的相似度值，避免无意义的阈值
    _, sim_list = get_sim(cand_qt_tran_sen2vec, init_qt_tran_sen2vec)
    f1_list = []
    thre_list = np.arange(min(sim_list), 1, (1 - min(sim_list)) / 20)
    for i in thre_list:
        result_wr, _ = get_sim(cand_qt_tran_sen2vec, init_qt_tran_sen2vec, thresold=i)

        # 4.计算F1值
        f1 = get_f1(result_wr, init_label, cand_label)
        print("the thresold is %f, the f1 is %f !" % (i, f1))
        f1_list.append(f1)

    print("the thresold is: ", thre_list)
    print("the f1_list is: ", f1_list)

    return max(f1_list)


@cost_time
def get_all_result(file_name, init_file, pattern_list, cut_list, remove_flag=True):
    f1_list = []
    for pattern in pattern_list:
        for cut in cut_list:
            print("**********************the %s and the cut is %s ! " % (pattern, str(cut)))
            f1 = get_one_result(file_name, init_file, pattern, cut=cut, remove_flag=remove_flag)
            f1_list.append(f1)

            print("********************************the %s and the cut is %s , "
                  "finished!!!!!!!!!!" % (pattern, str(cut)))

    return f1_list


if __name__ == "__main__":
    file_name = './data/人工标注数据集.txt'
    init_file = './data/query_pair2019-09-24.txt'
    # pattern_list = ['sa', 'wr', 'cw_wr']
    # pattern_list = ['qt']
    # cut_list = [True, False]
    # char_word2vec_file = ['./model/word2vec/sent_char_rem.txt_model_300.vector']
    # char_word_freque_file = './data/sent_char_rem/word_frequent.txt'
    # char_model_name = './model/word2vec/sent_char_rem.txt_model_300.model'
    # word_word2vec_file = ['./model/word2vec/sent_word_rem.txt_model_300.vector',
    #                       './model/word2vec/sent_word_rem.txt_model_300.model.wv.vectors.npy']
    # word_word_freque_file = './data/sent_word_rem/word_frequent.txt'
    # word_model_name = './model/word2vec/sent_word_rem.txt_model_300.model'
    # f1_list = get_all_result(file_name, init_file,
    #                          char_word2vec_file, char_word_freque_file, char_model_name,
    #                          word_word2vec_file, word_word_freque_file, word_model_name,
    #                          pattern_list, cut_list, remove_flag=True)

    pattern_list = ['sa', 'wr', 'cw_wr', 'qt', 'tr', 'qt_tr']
    cut_list = [True, False]
    f1_list = get_all_result(file_name, init_file, pattern_list, cut_list)
    print("the final result is: ", f1_list)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append('.')
sys.path.append('..')
import os
import jieba
from jieba.analyse import *
import multiprocessing
from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
from run_time import *
import threading

# 该文件主要用于生成词向量，对原始数据进行一系列的处理

# 1.解压：把.xml.bz2转化为.txt文件
@cost_time
def convert_txt(input_file):
    wiki = WikiCorpus(input_file, lemmatize=False, dictionary={})
    count = 0
    with open("./wiki.txt", 'w', encoding="utf-8") as f:
        for line in wiki.get_texts():
            print(line)
            line = ' '.join(line) + "\n"
            f.write(line)
            count += 1
    print("the %d articles have been writed" % count)


# 获取停词表
@cost_time
def get_stoplist(file_name):
    stop_list = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            stop_list.append(line.strip())

    return stop_list


# 数据清洗：只保留原始数据中的数字、中文和英文
def is_uchar(uchar):
    """判断一个unicode是否是汉字"""
    if u'\u4e00' <= uchar <= u'\u9fff':
        return True


def is_english(uchar):
    """判断是否是英文"""
    if 'a' <= uchar <= 'z' or 'A' <= uchar <= 'Z':
        return True
    return False


def is_number(uchar):
    """判断是否为数字"""
    if '0' <= uchar <= '9':
        return True
    return False


def pre_str(instr):
    """
       只保留字符串中的数字、英文和中文
       输入：字符串
       输出：清洗后的字符串
    """
    out_str = ''
    for uchar in instr:
        uchar = uchar.strip()
        if is_uchar(uchar) or is_english(uchar) or is_number(uchar):
            out_str = out_str + uchar.strip()
    return out_str


# 生成用于训练的2个字数据集，包含去除停词和未去除停词
@cost_time
def get_char(corpus_name, stop_file_name, output_name='char.txt', remove_flag=True):
    if output_name in os.listdir('../data'):
        os.remove('../data/' + output_name)

    output = open('../data/' + output_name, 'w', encoding='utf-8')
    stop_list = get_stoplist(stop_file_name)
    i = 0
    with open(corpus_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            line = line.strip('\n')
            # 清洗句子中的无用信息,只保留中文、英文和数字
            line = pre_str(line)
            if len(line) > 0:
                sentence = []
                if remove_flag:
                    sentence = [i.strip() for i in line.strip() if i not in stop_list]
                else:
                    sentence = [i.strip() for i in line.strip()]

                if sentence and len(sentence) > 1:
                    sentence = ' '.join(sentence)  # str
                    output.write(sentence)
                    output.write('\n')
                    i = i + 1
                    if i % 1000 == 0:
                        print("the %d sentences have been writted" % i)

    output.close()
    print("the %s has been created! the total number is %d" % (output_name, i))


# 生成分词后的用于训练的2个词数据集,包含去除停词和未去除停词
@cost_time
def get_word(corpus_name, stop_file_name, output_name='word.txt', remove_bool=True):
    if output_name in os.listdir('../data'):
        os.remove('../data/' + output_name)

    ff = open('../data/' + output_name, 'w', encoding='utf-8')
    i = 0
    stop_list = get_stoplist(stop_file_name)
    with open(corpus_name, 'r', encoding='utf-8') as f:
        for sentence in f:
            sentence = sentence.strip()
            # 清洗句子中的无用信息,只保留中文、英文和数字
            sentence = pre_str(sentence)

            if sentence and len(sentence) > 1:
                # 分词
                sentence = [segment.strip() for segment in
                            jieba.cut(sentence.strip(), cut_all=False) if len(segment) > 0]
                # 判断是否去除停词
                if remove_bool:
                    sentence = ' '.join([seg.strip() for seg in sentence if seg not in stop_list])
                else:
                    sentence = ' '.join([seg.strip() for seg in sentence])

                if sentence and len(sentence.strip()) > 0:
                    ff.write(sentence.strip())
                    ff.write('\n')
                    i += 1
                    if i % 10000 == 0:
                        print('已分词： %s个句子' % i)

    ff.close()
    print("the %s has been created! the total length is %d " % (output_name, i))


# TF-IDF提取高频词
# for keyword, weight in extract_tags('文本', withWeight=True, topK=100):
#     print('%s %s' % (keyword, weight))

@cost_time
def create_model(inp, outp1, outp2, size=300):
    # LineSentence预处理大文件
    print("start training the word model")
    model = Word2Vec(LineSentence(inp), size=size, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
    print("the model have been created")


@cost_time
def get_all_model(inp_list, num_list):
    path = 'word2vec'
    if path not in os.listdir('../model'):
        os.mkdir(os.path.join('../model/', path))

    for file in inp_list:
        inp = os.path.join('../data/', file)
        for num in num_list:
            model_name = '../model/word2vec/%s_model_%s.model' % (file, str(num))
            output_name = '../model/%s_model_%s.vector' % (file, str(num))
            print('%s %s' % (model_name, output_name))
            create_model(inp, model_name, output_name, size=num)
    print("%d model have been created! " % len(inp_list))


def get_word2vector(model_name, word):
    model = Word2Vec.load(model_name)
    vector = model[word]

    return vector


if __name__ == "__main__":
    # 1.转换为txt
    # 第一种方法
    # input_file = './zhwiki-latest-pages-articles.xml.bz2'
    # convert_txt(input_file)

    # 第二种方法
    # 利用https://github.com/attardi/wikiextractor抽取wiki语料，转换成txt
    # python3 WikiExtractor.py -b 1500M -o extracted zhwiki-latest-pages-articles.xml.bz2

    # opencc -i wiki.zh.text.txt -o test.txt -c t2s.json
    # opencc -i wiki_00 -o wiki.zh.txt -c t2s.json

    # 3.生成分词和未分词的文件
    # corpus_name = './data/test_wiki.txt'
    # stop_file_name = './data/stoplist.txt'
    # get_char(corpus_name,stop_file_name)
    # get_word(corpus_name,stop_file_name)

    # 4.生成词向量模型
    # MemoryError: Unable to allocate array with shape (667162, 1200) and data type float32
    # 参考：https: // blog.csdn.net /xovee / article / details / 101077022
    # inp_list = ['char.txt','word.txt']
    # num_list = [300,1200,2400]
    # get_six_model(inp_list,num_list)
    # inp = './data/word.txt'
    # output_1 = './model/word_model_1200.model'
    # output_2 = './model/word_model_1200.vector'
    # size = 1200
    # inp_list = ['sent_char_n.txt','sent_char_rem.txt','sent_word_n.txt','sent_word_rem.txt']
    # inp_list = ['sent_word_n.txt']
    # num_list = [300]
    # t = threading.Thread(target=get_all_model, args=(inp_list, num_list))
    # t.start()
    stop_list = get_stoplist(r"D:\毕业论文\sentence-embedding\data\stoplist.txt")
    print(stop_list)

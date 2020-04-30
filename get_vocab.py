
import numpy as np
import collections
from run_time import *
import threading
import os
import sys
import shutil
sys.path.append('.')

# 本段代码主要为了生成Quick-Thoughts Vectors所需要的词向量字典文件和词频文件

# 存在两个问题：
# 1.分词或者未分词的文件（词频最小为1）中得到的词数量与词向量不一致（最小词频为5.）
# 2.分词或者未分词的文件中得到的词是否存在词向量文件，即是否存在对应的词向量，
# 词频小于5的词语在词向量文件中没有对应的词向量
# 解决方案：直接利用word2vec的词向量文件生成词典和词向量


# 1. 从预处理后的文件中统计词频
@cost_time
def get_word_fre(file_name, output_name='word_frequent.txt'):
    if not os.path.exists(file_name):
        print("the %s not exists ! " % file_name)
        return None

    output_dir = os.path.join("./data/", file_name.split('/')[-1][:-4])
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)
    output_name = os.path.join(output_dir, output_name)
    print("the output_name is: ", output_name)

    wordcount = collections.Counter()
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            wordcount.update(line.split())
    i = 0
    with open(output_name, 'w', encoding='utf-8') as f:
        for word, value in wordcount.items():
            if word:
                f.write(word + ' ' + str(value) + '\n')
                i += 1
    print("the %d words have been computed the freque" % i)

    return wordcount


# 2.将词向量.model文件转化为字典和词向量
@cost_time
def get_vocab_embedding(file_name, word_count, vector_name,
                        vocab_file='vocab.txt',
                        emb_file='emb.txt',
                        max_length=20000):
    """利用词向量模型生成词典和词向量文件"""
    vector_file = os.path.join("./model/word2vec", vector_name)
    if not os.path.exists(vector_file):
        print("the %s not exists ! " % vector_file)
        return None

    output_dir = os.path.join("./data/", file_name.split('/')[-1][:-4])
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # 1.从词频字典中获取top—max_length的词
    words = list(word_count.keys())
    freqs = list(word_count.values())
    sorted_indices = np.argsort(freqs)[::-1]

    top_words = []
    top_freqs = []
    # 词向量文件会忽略词频小于5的词，将词频小于5的词变为UNK
    for index in sorted_indices:
        if word_count[words[index]] >= 5:
            top_words.append(words[index])
            top_freqs.append(freqs[index])

    # 2.获取模型文件中的字典：key:词，value：词向量
    word_emb = {}
    i = 0
    with open(vector_file, 'r', encoding='utf-8') as g:
        for line in g:
            if len(line.split()) > 2:
                word = line.split()[0]
                if word == 'UNNK':
                    print("UNNK in the vectorfile")
                    return None

                vector = [float(i) for i in line.split()[1:]]
                assert len(vector) == 300
                if word:
                    word_emb[word] = vector
                    i += 1

    print("the total words of %s is %d " % (vector_name, i+1))  #加上UNNK

    # 3.获取字典文件(每一行是一个词，包括UNK)和词向量文件
    final_words = ['UNNK']
    # quick_thoughts限定字典大小最多为20000，如果大于20000获取频率较高的前20000词
    final_words.extend(top_words[:max_length-1])

    vocab_file = os.path.join(output_dir, vocab_file)
    with open(vocab_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(final_words))
    print("the %d words have been writed in %s" %(len(final_words),vocab_file))

    vec_list = []
    j = 0
    for word in final_words:
        try:
            vec_list.append(word_emb[word])
            j += 1
        except Exception as e:
            print("error: ", e)
            print("the word not in the word2vec.model! "
                  "the word: %s, the count: %d " % (word, word_count[word]))
            continue

    print("the %d word vector been writted in %s" % (j, emb_file))

    emb_file = os.path.join(output_dir,emb_file)
    vec_list = np.array(vec_list)
    np.savetxt(emb_file, vec_list)
    print("the %s have been created!" % emb_file)

    a = np.loadtxt(emb_file)
    print(a.shape)
    print(type(a))


@cost_time
def get_file():
    file_list = ["./data/sent_word_n.txt", "./data/sent_word_rem.txt",
                 "./data/sent_char_n.txt", "./data/sent_char_rem.txt"]
    for file_name in file_list:
        print("**********************************************")
        try:
            wordcount = get_word_fre(file_name)
            vector_name = "%s_model_300.vector" % file_name.split('/')[-1]
            get_vocab_embedding(file_name, wordcount, vector_name)
            print("the %s have been created!" % file_name)
        except Exception as e:
            print(e)
            continue


# ********************第二次修改
@cost_time
def get_vocab(file_name):
    """
       从预处理后的文件中选择出字频或词频较高的生成字典，
       最多不超过20000个
       file_name: 预处理的文件名字
       return：字典，全是字或词
    """
    word_count = get_word_fre(file_name)
    # 排序后返回列表，列表中每个元素是（key, value)
    sorted_dic = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    output_path = os.path.join("./data/", file_name.split('/')[-1][:-4])
    output_file = os.path.join(output_path, 'vocab.txt')
    if 'vocab.txt' in os.listdir(output_path):
        os.remove(output_file)
        print("*******the old vocab.txt have been removed!!!!!!!!!*************")

    count = 1
    with open(output_file, 'w') as f:
        f.write('UNNK' + '\n')
        for key, value in sorted_dic:
            # 保证非空且词频大于等于5
            if key.strip() and value >= 5:
                f.write(key.strip() + '\n')
                count += 1
            if count >= 20000:
                break

    print("the vocab_size is: ", count)


@cost_time
def get_all_file():
    file_list = ["./data/new_sent_word_n.txt", "./data/new_sent_word_rem.txt",
                 "./data/new_sent_char_n.txt", "./data/new_sent_char_rem.txt"]
    for file_name in file_list:
        print("**********************************************")
        try:
            get_vocab(file_name)
            print("the %s have been created!" % file_name)
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    t = threading.Thread(target=get_all_file)
    t.start()

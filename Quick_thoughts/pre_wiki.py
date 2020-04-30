
import re
import threading
import sys
import pandas as pd

sys.path.append('..')
from run_time import *
from word_embedding import  get_char,get_word


# 将wiki数据转化为列表，包含1079821个词组
@cost_time
def process_wiki(file_name):
    with open(file_name,'r',encoding='utf-8') as f:
        title = ''
        i = 0
        articles = []
        para = []
        for line in f:
            # line = line.strip()
            print("line: ",line)
            if line == '</doc>':
                articles.append(para)
                para=[]
                continue
            elif 'url=' in line and 'title=' in line:
                # print("line: ",line)
                m = re.findall('(title=)"([\s\S]*)"', line)
                title = m[0][1]
                i += 1
                continue
            elif line == title:
                title = ''
                continue
            else:
                line = line.replace(' ','')
                if line:
                    para.append(line)
                    # print("para: ",para)
        # print("last_para: ",para)
        articles.append(para)
        print("%d articles have been processed!" % len(articles))
        # assert  i == len(articles)

        return  articles


# 分句并统计每个句子的长度
def split_sentence(articles,file_name):
    length_list = []
    f = open(file_name,'w',encoding='utf-8')
    for article in articles:
        for line in article:
            line = line.strip()
            line= line.replace(' ','')
            if line:
                m = re.findall('。',line)
                if len(m)>0:
                    sentence = line.split('。')
                    for sent in sentence:
                        sent = sent.strip('\n')
                        if sent:
                            length_list.append(len(sent))
                            f.write(sent)
                            f.write('\n')
    f.close()
    print("the %d  lines have been writted!" % len(length_list))
    length = pd.DataFrame(length_list,columns=['length'])
    print(length.describe())


def get_char_word():
    # 生成分割后的句子文件
    # file_name = "./data/wiki.zh.txt"
    output = './data/new_sentece.txt'
    # articles = process_wiki(file_name)
    # split_sentence(articles, output)

    # 生成未分词的去除停词和未去除停词的文件
    stop_file_name = './data/stoplist.txt'
    print("start get the sent_char.txt ")
    get_char(output, stop_file_name, 'sent_char_rem.txt',remove_flag=True)
    get_char(output, stop_file_name, 'sent_char_n.txt',remove_flag=False)

    # 生成分词后去除停词和未去除停词的文件
    print("start get the sent_word.txt ")
    get_word(output, stop_file_name, 'sent_word_rem.txt',remove_bool=True)
    get_word(output, stop_file_name, 'sent_word_n.txt',remove_bool=False)


if __name__ == "__main__":
    t = threading.Thread(target=get_char_word)
    t.start()

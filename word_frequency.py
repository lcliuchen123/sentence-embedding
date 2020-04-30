
import os
import threading
import sys
sys.path.append('.')
from run_time import  *
import pandas as pd

@cost_time
def get_word_frequency(corpus_file):
    """获取词频文件"""
    word_dic = {}

    if corpus_file not in os.listdir('./data'):
        return ValueError("the %s not in the directory " % corpus_file)

    corpus = os.path.join("./data/", corpus_file)

    with open(corpus,'r',encoding='utf-8') as f:
        try:
            for line in f:
                for word in line.split():
                    word_dic[word] = word_dic.get(word,0)+1
        except Exception as e:
            print(e)

    output = "./data/%s_frequency.txt" % corpus_file[:-4]
    with open(output,'w',encoding='utf-8') as f:
        for key,value in word_dic.items():
            print("%s %d" % (key,value))
            f.write("%s %d" % (key,value))
            f.write('\n')
    print("the %s have been created! " % corpus_file)
    return  word_dic

if __name__ == "__main__":
    corpus_name_list = ["sent_char_n.txt","sent_char_rem.txt","sent_word_n.txt","sent_word_rem.txt"]
    for corpus_name in corpus_name_list:
        t = threading.Thread(target=get_word_frequency,args=(corpus_name,))
        t.start()

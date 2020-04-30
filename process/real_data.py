
import  pandas as pd
import  threading
import sys
sys.path.append('.')
sys.path.append('..')
from process.get_predata import *
from jaccard import  *
import numpy as np
from run_time import *


# 查看真实数据集的长度
@cost_time
def get_length(file_name):
    short_text = open("../data/short_xaioai.txt",'w',encoding='utf-8')
    long_text = open("../data/long_xiaoai.txt",'w',encoding='utf-8')

    test_data = []
    lengt_list = []
    i = 0
    short_num = 0
    long_num = 0
    with open(file_name,'r',encoding='utf-8') as f:
        for line in f:
            lengt_list.append(len(line))
            if len(line)<10:
                short_text.write(line.strip()+'\n')
                short_num += 1
            else:
                long_text.write(line.strip()+'\n')
                long_num += 1
            test_data.append(line)
            i += 1
    short_text.close()
    long_text.close()

    # 635671 short texts, 364329 long texts
    print("%d short texts, %d long texts" % (short_num,long_num))
    length = pd.DataFrame(lengt_list)
    print(length.describe())
    print("total number of %s is %d" % (file_name,i))

    return test_data


# 按照重合度获取数据集
@cost_time
def get_test(num):
    """num: 表示需要处理的数据量"""
    test_file_name = '../data/xiaoai.txt'
    test_data = get_length(test_file_name)[:num]
    file_name = '../data/query_pair2019-09-24.txt'
    pre_sent = get_pre_sentence(file_name)
    pre = list(pre_sent.keys())
    overlap = []
    for test in test_data:
        t_overlap = []
        for p in pre:
            dis = edit_ditance(p,test)
            dis_rate = dis/max(len(p),len(test))
            # print("%s and %s: %f " %(test,p,dis_rate))
            t_overlap.append(dis_rate)
        overlap.append(t_overlap)

    overlap = np.array(overlap)
    print(overlap.shape)
    min_slap = np.min(overlap,axis=1) # 越小重合度越高
    print(min_slap)
    sort_slices = np.argsort(min_slap)
    board = int(len(sort_slices) * 0.2)
    print("the board is: %d " % board)
    low = open('../data/low_overlap.txt','w',encoding='utf-8')
    high = open('../data/high_overlap.txt','w',encoding='utf-8')

    for i in sort_slices[:board]:
        high.write(test_data[i].strip()+'\n')

    for j in sort_slices[-board:]:
        low.write(test_data[j].strip()+'\n')

    low.close()
    high.close()

    np.savetxt('../data/rate.txt',overlap)

    return overlap


if __name__ == "__main__":
    # file_name = '../data/xiaoai.txt'
    # file_name = '../data/query_pair2019-09-24.txt'
    # t = threading.Thread(target=get_pre_sentence,args=(file_name,))
    num = 10000
    t = threading.Thread(target=get_test,args=(num,))
    t.start()

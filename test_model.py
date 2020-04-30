
from run_time import *
from gensim.models import Word2Vec
import numpy as np
# import tensorflow as tf


# #5.测试
@cost_time
def test(model_name):
    en_wiki_word2vec_model = Word2Vec.load(model_name) # wiki.zh.text.model：模型名字
    testwords = ['苹果', '数学', '学术', '白痴', '篮球', '男人', '学习', '中国', '水池', '小狗']
    # testwords = ['中', '国', '学', '爱', '笑', '打', '酒', '水', '书', '光', '家', '男', '狗']
    for i in range(len(testwords)):
        try:
            res = en_wiki_word2vec_model.most_similar(testwords[i])
            print(testwords[i])
            print("res: ", res)
            vector = en_wiki_word2vec_model[testwords[i]]
            print(vector.shape)
        except:
            print("%s not in the vocab" % testwords[i])


if __name__ == "__main__":
    model_name = "./model/word2vec/sent_word_rem.txt_model_300.model"
    test(model_name)
    # test('./model/char_model_300.model')
    # print('*****************')
    # test('./model/char_model_1200.model')
    # with open("./data/char.txt",'r',encoding='utf-8') as f:
    #     for line in f:
    #         print(line)
    #         print(len(line))

    # file = r'F:\毕业论文\sentence-embedding\model\word2vec\sent_word_n_300.model.wv.vectors.npy'
    # w2v = np.load(file)
    # print(w2v)

    def dynamic_rnn(type='lstm'):
        x = np.random.randn(3,6,4)
        x[1,4:] = 0
        x_length = [6,4,6]
        rnn_hidden_size = 5
        if type == 'lstm':
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size,state_is_tuple=True)
        else:
            cell = tf.contrib.rnn.GRUCell(num_units=rnn_hidden_size)
        outputs,last_states = tf.nn.dynamic_rnn(
            cell = cell,
            dtype = tf.float64,
            sequence_length = x_length,
            inputs=x
        )

        with tf.Session() as session:
            # 初始化参数
            session.run(tf.global_variables_initializer())
            o1,s1 = session.run([outputs,last_states])
            print(np.shape(o1))
            print(o1)
            print(np.shape(s1))
            print(s1)
            # 保存了所有时刻的隐层状态
            # print([o1[0][-1],o1[1][-1],o1[2][-1]])
            print(s1[1])

    dynamic_rnn(type='GRU')
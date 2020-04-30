
import tensorflow as tf
from WR.wr_pre import *
from Quick_thoughts import input_ops
import numpy as np
import math


# 文件预处理与Quick-thoughts一致，(id, mask), Quick-thoughts的句子长度大于30会被截断，小于30不变，输入的句子长度不一致
# transformer的每个batch的长度可能存在差别

# 1.利用预训练或随机初始化词向量矩阵
# 保证词向量矩阵的顺序与字典中词的顺序一致
class Embedding(object):
    def __init__(self, emb_file_list=None, vocab=None, emb_matrix=None):
        self.emb_file_list = emb_file_list
        self.vocab = vocab
        self.emb_matrix = emb_matrix

    def get_vocab_emb(self):
        if self.emb_file_list:
            self.vocab, self.emb_matrix = getWordmap(self.emb_file_list)
        else:
            tf.logging.info("the file_list null, please input the correct file_list!!!")


# 2.编写位置编码
def get_position(max_seq_length, dim):
    """每个batch的长度不一样，位置编码自然也不一样，batch_size和seq_length可能都不一样"""
    pe = np.zeros((max_seq_length, dim))
    for i in range(max_seq_length):
        for j in range(dim):
            if j % 2 == 0:
                pe[i][j] = math.sin(i/(10000**(j/dim)))
            else:
                pe[i][j] = math.cos(i/(10000**((j-1)/dim)))
    tf.logging.info("*********************the position matrix have been created!!")
    tf.logging.info(pe)

    return pe


# 3.多头注意力层Q,K,V
def multi_attention(inputs, masks, num_head, type, mode, encode_output=None):
    """
    inputs: [batch_size,seq_length,num_units]
    masks: [batch_size,seq_length]
    num_head: the number of head in model'
    type:encode,decode,encode-decode三种注意力类型
    return:  [batch_size,seq_length,num_units]
    """
    min_num = -2**32 + 1
    emb_dim = inputs.get_shape().as_list()
    print("*********emb_dim: ", emb_dim)
    dim = emb_dim[-1]

    if dim % num_head:
        raise ValueError('please make the dim/d_head is the int!! ')

    # 1.初始化q,k,v矩阵，相当于添加一个全连接层，将输入的维度由[batch_size,seq_length,dim]变为[batch_size,seq_length,dim//num_head]
    q = tf.layers.dense(inputs, units=dim, use_bias=False)  # 全连接层,权重共享，对每个单词进行全连接，实现时所有位置单词是并行的
    k = tf.layers.dense(inputs, units=dim, use_bias=False)
    v = tf.layers.dense(inputs, units=dim, use_bias=False)
    tf.logging.info(q)

    # 只有编码器-解码器的权重的查询值与key,value不一样
    if type == 'encode-decode':
        k = encode_output  # 此时k和v是encode的输出，q是decode的输入
        v = encode_output

    # 切分为多头,q_:[batch_size*num_head,seq_length,dim/num_head]
    q_ = tf.concat(tf.split(q, num_head, axis=2), axis=0)
    k_ = tf.concat(tf.split(k, num_head, axis=2), axis=0)
    v_ = tf.concat(tf.split(v, num_head, axis=2), axis=0)
    tf.logging.info(q_)

    # scale q_
    depth = dim // num_head
    q_ *= (depth ** -0.5)

    # 计算q与k的相似度，sim:[batch_size*num_head,seq_length,seq_length]
    sim = tf.matmul(q_, k_, transpose_b=True)
    sim = sim / (k_.get_shape().as_list()[-1]**0.5)

    # 利用mask乘以一个特别小的数，避免填补的值对权重的影响。mask有padding mask和seq mask两种
    # 1.padding mask是句子长度不到固定长度的填补标记
    seq_length = tf.shape(masks)[1]
    masks = tf.to_float(masks)
    pad_mask = tf.tile(tf.expand_dims(masks, axis=1), [1, seq_length, 1])  # [batch_size, seq_length, seq_length]
    multi_pad_mask = tf.tile(pad_mask, [num_head, 1, 1])  # [batch_size*num_head, seq_length, seq_length]
    tf.logging.info(multi_pad_mask)

    # 2.seq mask是解码器中需要把t时刻后的单词进行覆盖,生成一个下三角
    seq_mask = tf.ones_like(masks, dtype=tf.float32)
    seq_mask = tf.linalg.LinearOperatorLowerTriangular(seq_mask).to_dense()  # 将张量转换为下三角矩阵
    multi_seq_mask = tf.tile(tf.expand_dims(seq_mask, axis=1), [1, seq_length, 1])
    multi_seq_mask = tf.tile(multi_seq_mask, [num_head, 1, 1])

    # encode只需要padding mask, decode需要padding mask 和 seq mask
    padding = tf.ones_like(sim) * min_num  # 创造一个与sim具有相同形状的张量，元素全为最小数
    if type == 'encode' or type == 'encode-decode':
        sim = tf.where(tf.equal(multi_pad_mask, True), sim, padding)  # 如果为真，就保留权重，否则利用最小数代替
    elif type == 'decode':
        # 如果相加可能存在1+0=1，导致部分掩盖被忽略
        # decode_mask = tf.add(multi_pad_mask, multi_seq_mask)

        # 相乘时只有两者里面都为1，才表示1，即没有被掩盖的部分
        decode_mask = multi_pad_mask * multi_seq_mask
        sim = tf.where(tf.equal(decode_mask, True), sim, padding)
    else:
        raise ValueError("please input encode, decode or encode-decode!!!")

    # 归一化，确保权重之和为1
    sim = tf.nn.softmax(sim)

    # dropout， 添加
    if mode == 'train':
        sim = tf.layers.dropout(sim, rate=0.3, training=True)

    # concat输出output
    output = tf.matmul(sim, v_)         # 输出output：[batch_size*num_head,seq_length,dim//num_head]
    output = tf.concat(tf.split(output, num_head, axis=0), axis=2)  # 输出output：[batch_size,seq_length,dim]

    # linear, 多头注意层的最后输出需要添加一个线性变换
    output = tf.layers.dense(output, units=dim, use_bias=False)

    return output


# 4.Layer norm
def layer_norm(inputs, eplsion=1e-8):
    """
    layer norm
        inputs: [batch_size,seq_length,num_units]
        eplsion：防止分母为0
        return:  [batch_size,seq_length,num_units]
    """
    para = inputs.get_shape().as_list()[-1]
    alpha = tf.Variable(tf.ones(para))
    beta = tf.Variable(tf.zeros(para))
    mean, var = tf.nn.moments(inputs, axes=[-1], keep_dims=True)
    norm = (inputs - mean) / ((var + eplsion) ** 0.5)

    # 数组中*是按照对应位置相乘的
    output = alpha * norm + beta  # 相当于alpha乘以[batch_size, seq_length], 然后乘以norm:[batch_size,seq_length,dim]

    return output


# 5.FFN
def forward_feed(inputs, num_units_list, mode='train'):
    """前馈全连接层FFN(x) = w_2*(max(0, w_1*x+b_1)) + b_2
    相当于两个大小为1的一维卷积层,二者选择一种方式实现即可
    inputs = word Embedding + position Embedding , [batch_size,seq_length,word_dim]
    num_units_list: contains two int number ,for example: [2048, 512]
    return: [batch_size,seq_length,word_dim]
    """
    # 1.全连接层实现
    inner = tf.layers.dense(inputs, units=num_units_list[0], activation=tf.nn.relu)

    # dropout, rate表示被丢弃的比例
    if mode == 'train':
        inner = tf.layers.dropout(inner, rate=0.3, training=True)

    output = tf.layers.dense(inner, units=num_units_list[1])  # 没有激活函数就是线性激活函数层

    # 2.卷积层实现
    # 相当于 [batch_size*sequence_length,num_units]*[num_units,ffn_dim]，在reshape成[batch_size,sequence_length,num_units]
    # paras = {"inputs": inputs, "kernel_size": 1, "filter": num_units_list[0], "activation": tf.nn.relu}
    # inner = tf.layers.conv1d(**paras)
    # paras = {"inputs": inner, "kernel_size": 1, "filter": num_units_list[1], "activation": tf.nn.relu}
    # output = tf.layers.conv1d(**output)

    return output


# 6.encoder
class Encoder(object):
    def __init__(self, emb, mask, num_head, num_units_list, mode):
        self.emb = emb    # batch_size,seq_length,dim
        self.num_head = num_head
        self.mask = mask
        self.num_units_list = num_units_list
        self.mode = mode

    def encoder(self):
        # 多头自注意力层
        multi_head_output = multi_attention(self.emb, self.mask, self.num_head, type='encode', mode=self.mode)

        # 残差连接
        v1_output = tf.add(multi_head_output, self.emb)

        # layer norm
        ln_output = layer_norm(v1_output)

        # 前馈全连接层
        ffn_output = forward_feed(ln_output, self.num_units_list, self.mode)

        # 残差链接
        v2_output = tf.add(ln_output, ffn_output)

        fin_ly_output = layer_norm(v2_output)

        return fin_ly_output


# 7.解码器 decoder
class Decoder(object):
    def __init__(self, decode_emb, num_head, num_units_list, decode_mask, mode):
        self.emb = decode_emb    # batch_size,seq_length,dim
        self.num_head = num_head
        self.num_units_list = num_units_list
        self.mask = decode_mask
        self.mode = mode

        """
        mask中的0不能直接参与计算，如果为0，经过softmax函数会变成1，可以用一个特别小的数代替
        解码器中t时刻之后的信息是无法获取的
        """

    def decode(self, encode_mask, encode_output, embedding):
        # masked decode self-attention
        decode_output = multi_attention(self.emb, self.mask, self.num_head, type='decode', mode=self.mode)

        # 残差链接
        decode_output = tf.add(decode_output, self.emb)

        # layer norm
        decode_ly_norm = layer_norm(decode_output)

        # encode_decode_attention
        encode_decode_output = multi_attention(decode_ly_norm, encode_mask, self.num_head, type='encode-decode',
                                               mode=self.mode, encode_output=encode_output)

        # 残差连接
        encode_decode_output = tf.add(encode_decode_output, decode_ly_norm)

        # layer norm
        encode_decode_ly_norm = layer_norm(encode_decode_output)

        # 前馈全连接层
        ffn_output = forward_feed(encode_decode_ly_norm, self.num_units_list, self.mode)

        # 残差连接
        ffn_output = tf.add(ffn_output, encode_decode_ly_norm)

        # layer norm
        ffn_ly_norm = layer_norm(ffn_output)

        return ffn_ly_norm


# 8. 标签平滑
def label_smoothing(inputs, epsilon=0.1):
    """
    标签平滑
    """
    v = inputs.get_shape().as_list()[-1]
    output = (1-epsilon)*inputs + epsilon / v

    return output


class s2v_model(object):
    def __init__(self, mode, batch_size, input_file_pattern, uniform_init, emb_file, num_head, num_units_list,
                 num_input_reader_threads, input_queue_capacity, vocab_size=9592, encode_dim=300, seq_length=30,
                 shuffle_input_data=False, input_reader=None, input_queue=None):
        if mode not in ['train', 'encode']:
            raise ValueError("Please input the train or encode !!!!")
        self.reader = input_reader if input_reader else tf.TFRecordReader()
        self.input_queue = input_queue
        self.shuffle_input_data = shuffle_input_data
        self.batch_size = batch_size
        self.input_file_pattern = input_file_pattern
        self.num_input_reader_threads = num_input_reader_threads
        self.input_queue_capacity = input_queue_capacity
        self.mode = mode
        self.encode_ids = None
        self.encode_mask = None
        self.decode_ids = None
        self.decode_mask = None
        self.trans_vec = None
        self.total_loss = None
        self.uniform_initer = tf.random_uniform_initializer(minval=-uniform_init, maxval=uniform_init)
        self.emb_file = emb_file
        self.num_head = num_head  # 多头
        self.num_units_list = num_units_list  # 全连接层的单元数
        self.encode_dim = encode_dim
        self.seq_length = seq_length

        self.vocab_size = vocab_size
        self.init = None
        self.embedding = None  # 训练或预测时用到的词向量，在最后的线性输出中也用到了

    def build_inputs(self):
        """
        type:表明是encode还是decode
        """
        decode_ids = None
        decode_mask = None
        if self.mode == 'encode':
            encode_ids = tf.placeholder(tf.int64, shape=(None, None), name='encode_ids')
            encode_mask = tf.placeholder(tf.int8, shape=(None, None), name='encode_mask')
        elif self.mode == 'train':
            input_queue = input_ops.prefetch_input_data(
                self.reader,
                self.input_file_pattern,
                shuffle=self.shuffle_input_data,
                capacity=self.input_queue_capacity,
                num_reader_threads=self.num_input_reader_threads)
            print("input_queue: ", input_queue)
            print("input_queue_size: ", input_queue.size())
            encode_serialized = input_queue.dequeue_many(self.batch_size)
            encode = input_ops.parse_example_batch(encode_serialized)
            tf.logging.info(encode)

            encode_ids = encode.ids[:-1]
            encode_mask = encode.mask[:-1]
            decode_ids = encode.ids[1:]
            decode_mask = encode.mask[1:]

            # encode_ids = padding_batch(encode.ids[:-1], self.seq_length, self.batch_size)
            # encode_mask = padding_batch(encode.mask[:-1], self.seq_length, self.batch_size)
            # decode_ids = padding_batch(encode.ids[1:], self.seq_length, self.batch_size)
            # decode_mask = padding_batch(encode.mask[1:], self.seq_length, self.batch_size)
        else:
            raise ValueError('please input the type of train or encode!!!')

        return encode_ids, encode_mask, decode_ids, decode_mask

    def get_position(self, max_seq_length, dim):
        """pe是一个seq_length*dim的矩阵，表明句子中每个单词每个维度的位置编码"""
        pe = np.zeros((max_seq_length, dim))
        for i in range(max_seq_length):
            for j in range(dim):
                if j % 2 == 0:
                    pe[i][j] = math.sin(i/(10000**(j/dim)))
                else:
                    pe[i][j] = math.cos(i/(10000**((j-1)/dim)))
        tf.logging.info("**the position matrix have been created!! And the shape is: %d, %d ", max_seq_length, dim)
        tf.logging.info(pe)

        return pe

    def build_emb(self, mask, encode_type):
        """词向量+位置编码"""
        length = tf.to_int32(tf.reduce_sum(mask, 1))  # 所有样本的长度列表
        max_length = tf.reduce_max(length)
        # tf.logging.info("*************the max length *******")

        encode_emb = []
        with tf.variable_scope(encode_type, reuse=tf.AUTO_REUSE):
            if self.mode == 'train' or self.mode == 'encode':
                if self.emb_file:
                    word_vecs = np.loadtxt(self.emb_file)
                    vocab_size = word_vecs.shape[0] + 2  # padding, UNNK
                    word_dim = word_vecs.shape[1]
                    tf.logging.info("the vocab_size is %d, the word_dim is %d ", vocab_size, word_dim)
                    self.vocab_size = vocab_size

                    assert self.encode_dim == word_dim

                    word_emb = tf.get_variable(name='word_embedding', shape=[vocab_size, word_dim], trainable=False)
                    embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, word_dim])
                    embedding_init = word_emb.assign(embedding_placeholder)

                    # 对于未登录词，随机初始化,对应的索引为1
                    rand = np.random.rand(1, word_dim)
                    # 对于padding的词语直接赋值为0,对应的索引为0
                    pad_word2vec = np.zeros((1, word_dim))

                    word_init = np.concatenate((pad_word2vec, rand, word_vecs), axis=0)
                    tf.logging.info("***********the length of word vector is %d *********", word_init.shape[0])
                    # self.embedding = word_init
                    self.init = (embedding_init, embedding_placeholder, word_init)
                else:
                    # 没有预先训练好的词向量文件，把字典的大小设为字典的长度
                    # self.vocab_size = 20002
                    tf.logging.info("the word embedding file not exist, so the word embedding is random initialized!!,"
                                    "the vocab_size is %d , and the encode dim is %d !",
                                    self.vocab_size, self.encode_dim)
                    word_emb = tf.get_variable(name='word_embedding', shape=[self.vocab_size, self.encode_dim],
                                               initializer=tf.random_normal_initializer(0., self.encode_dim ** -0.5))
                self.embedding = word_emb  # 词向量
                if encode_type == 'encode':
                    encode_word_emb = tf.nn.embedding_lookup(word_emb, self.encode_ids)  # 词向量
                elif encode_type == 'decode':
                    encode_word_emb = tf.nn.embedding_lookup(word_emb, self.decode_ids)  # 词向量
                else:
                    raise ValueError("************Please input the correct type: encode or decode!********* ")

                # 对随机生成的词向量进行标准化
                encode_word_emb *= (self.encode_dim ** 0.5)

                tf.logging.info("*******the shape of word embedding*************")
                tf.logging.info(encode_word_emb)

                tf.logging.info("*********************the position embedding******")
                # pe = self.get_position(self.seq_length, word_dim)

                pe = self.get_position(self.seq_length, self.encode_dim)
                pe = tf.constant(pe, dtype=tf.float32, name="pe")
                shape = tf.shape(encode_word_emb)
                batch_length = shape[0]
                seq_length = shape[1]

                position_index = tf.tile(tf.expand_dims(tf.range(seq_length), 0), [batch_length, 1])
                position_enc = tf.nn.embedding_lookup(pe, position_index)

                # 添加padding，padding的部分的词向量为0
                position_enc = tf.where(tf.equal(encode_word_emb, 0), encode_word_emb, position_enc)
                position_enc = tf.to_float(position_enc)
                tf.logging.info(position_enc)
                encode_emb = tf.add(encode_word_emb, position_enc)  # [batch_size,seq_length,dim]
            else:
                raise ValueError("please input the correct mode: train or encode !!!")
        tf.logging.info(encode_emb)
        tf.logging.info("*****************the word embedding and position embedding have been created !!!**********")
        return encode_emb

    def build_encoder(self, num_ceng=2):
        # 原论文默认6层，本文选择2层
        # 多头注意力层
        # 全连接层
        # add norm层
        encode_output = None
        self.encode_ids, self.encode_mask, self.decode_ids, self.decode_mask = self.build_inputs()
        self.encode_emb = self.build_emb(self.encode_mask, 'encode')
        inputs = self.encode_emb
        tf.logging.info("the inputs: ")
        tf.logging.info(inputs)

        for i in range(num_ceng):
            encode = Encoder(inputs, self.encode_mask, self.num_head, self.num_units_list, self.mode)
            encode_output = encode.encoder()
            inputs = encode_output
        tf.logging.info("************the %d ceng encoder have been created !! ", num_ceng)
        return encode_output

    def build_decoder(self, encode_output, num_ceng=2):
        decode_output = None
        self.decode_emb = self.build_emb(self.decode_mask, 'decode')
        inputs = self.decode_emb
        for i in range(num_ceng):
            decode = Decoder(inputs, self.num_head, self.num_units_list, self.decode_mask, self.mode)
            decode_output = decode.decode(self.encode_mask, encode_output, self.embedding)
            inputs = decode_output
        tf.logging.info("************the %d ceng decoder have been created !! ", num_ceng)
        return decode_output

    def build_loss(self):
        encode_output = self.build_encoder()
        tf.logging.info("**************the encoder output")
        tf.logging.info(encode_output)

        decode_output = self.build_decoder(encode_output)

        # output: linear+softmax
        weight = tf.transpose(self.embedding)  # [dim,vocab_size]

        # tf.einsum矩阵乘法，可以【1，2，4】*【4，3】，tf.matmul不允许这样做矩阵乘法，与刚开始的嵌入共享权重矩阵
        linear_output = tf.einsum('bsd,dv->bsv', decode_output, tf.cast(weight, tf.float32))   # [batch_size,seq_length,vocab_size]
        logits = tf.nn.softmax(linear_output)

        y_ = tf.one_hot(self.decode_ids, depth=self.vocab_size)
        tf.logging.info(y_)
        y_real = label_smoothing(y_)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_real, logits=logits)
        nopadding_loss = tf.matmul(loss, tf.cast(self.decode_mask, tf.float32), transpose_b=True)
        total_loss = tf.reduce_sum(nopadding_loss)/tf.cast(tf.reduce_sum(self.decode_mask), tf.float32)
        tf.summary.scalar("total_loss", total_loss)
        self.total_loss = total_loss
        tf.logging.info("the loss function have been created !!!")

    def build_encode(self):
        encode_output = self.build_encoder()
        tf.logging.info("**************the encoder output in encode function")
        tf.logging.info(encode_output)
        return encode_output

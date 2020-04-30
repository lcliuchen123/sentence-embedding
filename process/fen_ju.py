import sys
sys.path.append('.')
sys.path.append('..')

from word_embedding import *
from run_time import *
from process.pi_pei import *

# 该文件的主要作用是生成句向量生成算法的训练样本
# 输入：wiki.zh.txt其实已经被转换成一段一段的句子。
# 1.分句后，可能还有一些噪声，比如“！1，！2”.
# 2.维基百科中的一些噪声：colspan = 2 style = "text-align:left;" | ↓
# 3.部分句子只有一个标点符号，可能是包含有空格，所以没被过滤掉
# 4.部分语句冒号：后面的没有接到一块
# 5.部分语句中同一个事物有多种表达方式，利用正则匹配


"""第一次的分句代码"""


# 生成的数据集中存在空格，因为停词表没有加入空格，
# 进行训练时应该去除中间的空格,sentence.split()不会出现多余的空格。
# 按照句号分句，会忽略信息，
# 百科数据中会对一些词条进行解释，词条中存在冒号，将冒号的语句添加到下一句前面。


# 先将维基百科中的句子进行分句,然后在去除停词，进行分词或者不分词
def fen_ju(data_file, output_name='new_sentence.txt'):
    if not data_file:
        raise ValueError("please input the data_file")

    i = 0
    g = open('./data/' + output_name, 'w', encoding='utf-8')
    with open(data_file, "r", encoding='utf-8') as f:
        last = ''
        for article in f:
            sentence_length = re.findall('。', article)
            print("句子长度：", len(sentence_length))
            if len(sentence_length) > 0:
                if last:
                    article = last + article
                    last = ''
                sentence_list = article.split('。')
                for sentence in sentence_list:
                    sentence = sentence.strip('\n')
                    if sentence:
                        g.write(sentence)
                        g.write('\n')
                        i += 1
            # else:
            #     if article[:-1] == '：' and 'url' not in article:
            #         last += article

    g.close()
    print("the %s have been created! the total length is %d" % (output_name, i))


# 生成分句或者不分句的文件
@cost_time
def get_result():
    data_file = "./data/wiki.zh.txt"  # 15414839行
    fen_ju(data_file)
    stop_file_name = './data/stoplist.txt'
    print("start get the sent_char_n.txt ")
    get_char('./data/new_sentence.txt', stop_file_name, 'sent_char_n.txt', remove_flag=True)
    print("start get the sent_word_n.txt ")
    get_word('./data/new_sentence.txt', stop_file_name, 'sent_word_n.txt', remove_bool=True)


"""第二次的分句代码"""


def __merge_symmetry(sentences, symmetry=('“', '”')):
    """合并对称符号，如双引号
    可能双引号中语句过长"""
    effective_ = []
    merged = True
    for index in range(len(sentences)):
        if symmetry[0] in sentences[index] and symmetry[1] not in sentences[index]:
            merged = False
            effective_.append(sentences[index])
        elif symmetry[1] in sentences[index] and not merged:
            merged = True
            effective_[-1] += sentences[index]
        elif symmetry[0] not in sentences[index] and symmetry[1] not in sentences[index] and not merged:
            effective_[-1] += sentences[index]
        else:
            effective_.append(sentences[index])

    return [i.strip() for i in effective_ if len(i.strip()) > 0]


def to_sentences(paragraph):
    """由段落切分成句子，英文句号可能会切分数字，比如112.34亿元"""
    sentences = re.split(r"(？|。|！|!|\…\…)", paragraph)
    print(type(sentences))
    sentences.append("")
    print(sentences)
    # 将切分后的标点符号和句子合为一个字符串
    sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2])]
    print("sentences: ", sentences)
    sentences = [i.strip() for i in sentences if len(i.strip()) > 0]

    for j in range(1, len(sentences)):
        # 如果双引号被分到下一句，进行调整
        if sentences[j][0] == '”':
            print(sentences[j])
            sentences[j - 1] = sentences[j - 1] + '”'
            sentences[j] = sentences[j][1:]

        # 处理冒号,把下一句加到冒号后面
        if sentences[j-1][-1] == ":":
            sentences[j-1] += sentences[j]
            sentences[j] = ''

    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    print(sentences)

    return __merge_symmetry(sentences)


def split_para(data_file, output_name='new_sentence.txt'):
    if not data_file:
        raise ValueError("please input the data_file")
    if output_name in os.listdir("../data"):
        os.remove('../data/' + output_name)

    i = 0
    g = open('../data/' + output_name, 'w', encoding='utf-8')
    with open(data_file, "r", encoding='utf-8') as f:
        # 标签后有三行，例如：
        # </doc>
        # <doc id="70" url="https://zh.wikipedia.org/wiki?curid=70" title="天文学">
        # 天文学
        # 这些内容不需要
        count = 0
        for line in f:
            line = line.strip()
            line = line.strip('\n')
            # print("line: ", line)
            if line:
                if line == "</doc>":
                    count = 1
                    continue
                elif 0 < count < 3:
                    count += 1
                    continue
                # 保证句子中至少有两个字
                elif len(line) > 1 and count > 2:
                    count += 1
                    sentences = to_sentences(line)
                    print("sentences: ", sentences)
                    for sentence in sentences:
                        print("句子长度：", len(sentence))
                        sentence = sentence.strip()
                        if sentence:
                            # 处理繁转简后的同一事物的不同叫法问题
                            sentence = pre_line(sentence).strip()
                            g.write(sentence)
                            g.write('\n')
                            i += 1
                else:
                    continue

    g.close()
    print("the %s have been created! the total length is %d" % (output_name, i))


@cost_time
def write_file():
    data_file = "../data/wiki.zh.txt"  # 15414839行
    split_para(data_file)
    stop_file_name = '../data/stoplist.txt'
    remove_list = [True, False]
    for remove_flag in remove_list:
        print("start get the sent_char_rem.txt and sent_char_n.txt ")
        if remove_flag:
            word_file_name = "new_sent_word_rem.txt"
            char_file_name = "new_sent_char_rem.txt"
        else:
            word_file_name = "new_sent_word_n.txt"
            char_file_name = "new_sent_char_n.txt"

        get_char('../data/new_sentence.txt', stop_file_name, char_file_name, remove_flag=remove_flag)
        print("start get the sent_word_rem.txt and sent_word_n.txt")
        get_word('../data/new_sentence.txt', stop_file_name, word_file_name, remove_bool=remove_flag)


if __name__ == "__main__":
    t = threading.Thread(target=write_file)
    t.start()

    # test
    # para = '我心里暗笑他的迂；他们只认得钱，托他们只是白托!' \
    #        '而且我这样大年纪的人，难道还不能料理自己么？' \
    #        '唉，我现在想想，那时真是太聪明了!' \
    #        '我说道：“爸爸，你走吧。”' \
    #        '他往车外看了看说：“我买几个橘子去。你就在此地，不要走动。”' \
    #        '我看那边月台的栅栏外有几个卖东西的等着顾客。' \
    #        '走到那边月台，须穿过铁道，须跳下去又爬上去。'
    # sentences = to_sentences(para)
    # print("result: ", sentences)





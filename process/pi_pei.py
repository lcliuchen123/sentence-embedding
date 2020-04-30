#!/usr/bin/env/python3
# -*- coding: utf-8 -*-

"""
@Time    : 2020/4/22 9:31
@Author  : Chen Liu
@FileName: pi_pei.py
@Software: PyCharm
"""

import re


# 问题1：部分zh-cn后面没有分号
# 问题2：部分zh-cn与zh-hans同时出现,只选择其中的一个
# 问题3：部分语句中有多个满足条件的字符串需要匹配
# 问题4：部分语句满足的字符串里面为空，比如-{}-
def pre_line(line):
    """
       中文维基百科繁转简后，会出现多种术语（同一事物的不同叫法，实际应用中只需要保留一种）
       一般简体中文的限定词是 zh-hans 或 zh-cn
       例如：
           输入：同时还有许多参与-{zh-cn:自行车运动; zh-hk:踩单车运动; zh-tw:脚踏车运动;}-的民众
           输出：同时还有许多参与自行车运动的民众。
    """
    # 第一步找出满足条件的字符串，例如-{}-.
    old_str = re.findall(r'-\{.*?;?\}-', line)
    print("old_str: ", old_str)

    # new_str = re.findall(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-', line)
    # print("new_str: ", new_str)

    # 判断是否为空
    if old_str:
        for i in range(len(old_str)):
            print(i)
            # 查找每个满足条件的字符串需要被替换的新字符串，如果无法找到满足条件的字符串，则用‘’代替
            new_i = re.match(r'-\{.*?(zh-hans|zh-cn):([^;]*?)(;.*?)?\}-', old_str[i])

            if not new_i:
                new_str = ''
            else:
                new_str = new_i.group(2)
            print("new_str: ", new_str)

            line = line.replace(old_str[i], new_str)
            # print(line)

    return line


if __name__ == "__main__":
    s = "同时还有许多参与-{}-的民众和" \
        "-{zh-cn:自行车运动; zh-hk:踩单车运动; zh-tw:脚踏车运动;}-的人"
    sentences = '随著-{zh-hans:社会媒体; zh-cn:社交媒体; zh-tw:社群媒体; zh-hk:社群媒体;}-的快速发展，' \
                '中国在2015年时用户于-{zh-hans:数字媒体; zh-hk:数码媒体; zh-tw:数位媒体;}-的使用时长首次超过传统媒体'
    sent = "因不符合中国的网路安全法，这些网站遭中国政府相关的网络安全管理中心屏蔽，中国大陆的网民亦无法访问这些网站；" \
           "而部分网站如-{zh-tw:Bing;zh-hk:Bing;zh-cn:必应}-、Flipboard等则事前接受审查标准，再推出特别的中国地区服务。"

    sentence = ['莲雾（学名：），-{zh-hans:新加坡和马来西亚一带叫做水蓊; zh-hant:新加坡及马来西亚一带称为水蓊（Jambu Air);'
                ' zh-sg:台湾称作莲雾;zh-my:台湾称作莲雾}-，又名天桃，别名辇雾、琏雾、爪哇蒲桃、洋蒲桃，是桃金娘科的常绿小乔木。',
                '因其果实长得像铃铛，亦称为bell-fruit。',
                '原产于马来群岛，在马来西亚、印尼、菲律宾和台湾普遍栽培，是一种主要生长于热带的水果。', '']

    for line in sentence:
        line = pre_line(line)
        print(line)

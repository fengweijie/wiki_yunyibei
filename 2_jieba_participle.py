#!/usr/bin/env python
# -*- coding: utf-8  -*-
# 逐行读取文件数据进行jieba分词

import jieba
import jieba.analyse
import jieba.posseg as pseg  # 引入词性标注接口
import codecs, sys
import re


def preprocess_word(word):
    # if word in stopwords:
    #     return ''
    # 去除标点
    word = word.strip(
        '\'"?!,.():;？！，。…“”（）：；<>《》/ 、rn【】[]|~#%&*br = μ   α   θ   η   μ   α   τ   ι   κ   ά m   á   t h   ē   m a')
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def preprocess_sentence(tweet):
    processed_tweet = []
    # Convert to lower case转换成小写
    tweet = tweet.lower()
    # Replaces URLs with the word URL去除URL
    tweet = re.sub(r'((www\.[\S]+)|(https?://[\S]+))', ' URL ', tweet)
    # Strip space, " and ' from tweet
    tweet = tweet.strip(' "\'')
    words = jieba.cut(tweet)
    for word in words:
        word = preprocess_word(word)
        if len(word) != 0:
            processed_tweet.append(word)
    if len(processed_tweet) == 0:
        print(tweet)
        return ' '.join(['很好'])
    else:
        return ' '.join(processed_tweet)


if __name__ == '__main__':
    f = codecs.open('./wiki.txt', 'r', encoding='utf8')
    target = codecs.open('wiki.seg.txt', 'w', encoding='utf8')
    print('open files.')

    lineNum = 1
    line = f.readline()
    while line:

        seg_list = preprocess_sentence(line)
        line_seg = ' '.join(seg_list)
        target.writelines(line_seg)
        lineNum = lineNum + 1
        line = f.readline()
        if (lineNum % 500 == 0):
            print('---processing ', lineNum, ' article---')

    print('well done.')
    f.close()
    target.close()

# -*-coding:utf-8-*-
import re
import sys
from nltk.stem.porter import PorterStemmer
import jieba
import pandas as pd

STOP_WORDS_FILE = 'stopwords/new_stopwords.txt'


def preprocess_word(word):
    # if word in stopwords:
    #     return ''
    # 去除标点
    word = word.strip('\'"?!,.():;？！，。…“”（）：；<>《》/ 、rn【】[]|~#%&*br')
    # Remove - & '
    word = re.sub(r'(-|\')', '', word)
    return word


def preprocess_sentence(tweet, label):
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


def preprocess_csv(csv_file_name, processed_file_name, test_file):
    raw_data = pd.read_csv(csv_file_name)
    sentence = []
    for d, label in zip(raw_data['Discuss'].values, raw_data['Id'].values):
        processed_sentence = preprocess_sentence(d, label)
        sentence.append(processed_sentence)
    if test_file:
        res = pd.DataFrame({'Id': raw_data['Id'], 'Discuss': sentence})
    else:
        res = pd.DataFrame({'Id': raw_data['Id'], 'Discuss': sentence, 'Score': raw_data['Score']})
    res.to_csv(processed_file_name, index=False)
    # print(res)
    print(res.shape)


test_file = True
csv_file_name = './input/predict_second.csv'
processed_file_name = './cache/predict-processed.csv'
preprocess_csv(csv_file_name, processed_file_name, test_file)

test_file = False
csv_file_name = './input/train_second.csv'
processed_file_name = './cache/train-processed.csv'
preprocess_csv(csv_file_name, processed_file_name, test_file)
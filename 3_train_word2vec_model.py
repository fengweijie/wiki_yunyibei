#!/usr/bin/env python
# -*- coding: utf-8  -*-
#使用gensim word2vec训练脚本获取词向量

import warnings
from gensim.models import word2vec
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')# 忽略警告
import multiprocessing

from gensim.models import Word2Vec

senteces = word2vec.Text8Corpus('./wiki.seg.txt')
model = Word2Vec(senteces,min_count=5,size = 128,window=6,workers=multiprocessing.cpu_count())

print(model.most_similar('皇'))

outp1 = 'wiki.zh.text.model'
model.save(outp1)
model.wv.save_word2vec_format('char_vector_128.txt', binary=False)

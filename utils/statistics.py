import os
import json
from collections import Counter
from matplotlib import pyplot as plt

data_path = '/home/codes/pytorch-acnn-model/attention/train.txt'

f = open(data_path, 'r', encoding='utf-8')

data = [i.strip().split(' ') for i in f.readlines()]
data_words = [i[5:] for i in data]
# 词语的不同个数
data_words_set = set([j for i in data_words for j in i])
data_words_counter = Counter([j for i in data_words for j in i])
data_words_counter_sorted = sorted(data_words_counter.items(), key=lambda x:x[1], reverse=True)
print(json.dumps(data_words_counter_sorted))
print("不同words个数为：", len(data_words_set))

# 位置的不同个数
data_pos = [i[:5] for i in data]
data_pos = [[int(i) for i in j] for j in data_pos]
head_head_sub_pos = set([abs(i[1] - i[3]) for i in data_pos])
tail_tail_sub_pos = set([abs(i[2] - i[4]) for i in data_pos])
print('头到头的相对距离个数', len(head_head_sub_pos),
      '尾到尾的相对距离个数', len(tail_tail_sub_pos))


import torch
import numpy as np
import torch.nn as nn

#

class AcnnEmbedding:

    def __init__(self, word_number, demension, pre_embedding_path=None):
        self.word_number = word_number
        self.demension = demension
        self.pre_embedding_path = pre_embedding_path
        if self.pre_embedding_path:
            self.pre_embedding_data = self.load_pre_embedding()
        else:
            self.pre_embedding_data = None

    def init_embedding(self):
        # 初始化嵌入层
        if self.pre_embedding_path:
            embedding_layer = nn.Embedding(len(self.pre_embedding_data),
                                           len(self.pre_embedding_data[0]))
            embedding_layer.weight.data.copy_(torch.from_numpy(self.pre_embedding_data))
        else:
            embedding_layer = nn.Embedding(self.word_number, self.demension)
        return embedding_layer

    def load_pre_embedding(self):
        # 导入预训练词向量
        data = open(self.pre_embedding_path, 'r', encoding='utf-8').readlines()
        data = [i.strip().split() for i in data]
        data = [[float(j) for j in i] for i in data]
        return np.array(data)

if __name__ == '__main__':
    Em_word = AcnnEmbedding(100, 50, '/home/codes/pytorch-acnn-model/attention/embeddings.txt')
    word_layer = Em_word.init_embedding()
    print(word_layer)

    Em_pos = AcnnEmbedding(54, 50, '')
    pos_layer = Em_pos.init_embedding()
    print(pos_layer)



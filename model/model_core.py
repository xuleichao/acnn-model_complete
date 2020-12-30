import torch
import numpy as np
import torch.nn as nn
from model.embedding import AcnnEmbedding
from model.window_slide_utils import window_slide
from attention.first_attention import diag_array_init, A_matrix_construct
from model.cnn import acnn_layer

pre_w2v = '/home/codes/pytorch-acnn-model/attention/embeddings.txt'

class Acnn(nn.Module):

    def __init__(self):
        super(Acnn, self).__init__()
        # embending
        self.words_embedding = AcnnEmbedding(13000, 50,
                                             pre_embedding_path=pre_w2v).init_embedding()
        self.words_embedding.weight.requires_grad=False
        self.pos_embedding = AcnnEmbedding(56, 50).init_embedding()
        # attention first

        # res = A_matrix_construct(head_matrix, end_matrix)
        # cnn
        # 输出16通道，kenel:3 * 3
        self.cnn = acnn_layer(150, 16, 3)
        # pool attention
    def _A_matrix_construct(self, startpos_list):
        startpos_list_embedding = self.words_embedding(startpos_list)
        startpos_list_embedding_rsp = startpos_list_embedding.repeat(1, 3).reshape(2, -1, 50)
        return startpos_list_embedding_rsp

    def _R_matrix_construct(self, start, end):
        R = []
        for i in range(start.shape[0]):
            start_i = start[i]
            end_i = end[i]
            R_i = A_matrix_construct(start_i, end_i)
            R.append(R_i.tolist())
        return torch.from_numpy(np.array(R, dtype='float32'))

    def forward(self, sent_x, pos_left, pos_right, head_startpos_list, end_startpos_list, y):
        word_embedding = self.words_embedding(sent_x)
        pos_left_embedding = self.pos_embedding(pos_left)
        pos_right_embedding = self.pos_embedding(pos_right)

        # 构建A矩阵
        start_e_A = self._A_matrix_construct(head_startpos_list).bmm(word_embedding.reshape(2, 50, 3))
        end_e_A = self._A_matrix_construct(end_startpos_list).bmm(word_embedding.reshape(2, 50, 3))
        word_embedding = (start_e_A.mul(end_e_A)).bmm(word_embedding)
        net_embedding = torch.cat((word_embedding, pos_left_embedding, pos_right_embedding), 2)
        # CNN输出结果
        cnn_result = self.cnn(net_embedding.permute(0, 2, 1))
        # 获得cnn compose 矩阵 R
        R = self._R_matrix_construct(start_e_A, end_e_A)
        cnn_result.bmm(R.reshape(2, 1, 3))
        print(cnn_result)
        print(cnn_result.shape)

if __name__ == '__main__':
    pos_embedding_id_int = dict(enumerate(np.arange(-28, 28, 1)))
    pos_embedding_id_int.update({56: 999})
    pos_embedding_int_id = {j: i for i, j in pos_embedding_id_int.items()}

    M = Acnn()
    res = M.forward(torch.from_numpy(np.array([[1,2,3], [2,1,3]])),
                    torch.from_numpy(np.array([[1, 2, 0], [2, 2, 2]])),
                    torch.from_numpy(np.array([[2, 2, 0], [2, 2, 2]])),
                    torch.from_numpy(np.array([0, 1])),
                    torch.from_numpy(np.array([0, 1])),
                    torch.from_numpy(np.array([1,2])))

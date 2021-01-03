import torch
import numpy as np
import torch.nn as nn
from model.embedding import AcnnEmbedding
from torch.autograd import Variable
from model.window_slide_utils import window_slide
from attention.first_attention import diag_array_init, A_matrix_construct
from model.cnn import acnn_layer

pre_w2v = '/home/codes/pytorch-acnn-model/attention/embeddings.txt'
pre_w2v = r"F:\1Yong\codes\acnn-model-complete\datasets\embeddings.txt"

class Acnn(nn.Module):

    def __init__(self):
        super(Acnn, self).__init__()
        # embending
        self.words_embedding = AcnnEmbedding(13000, 50,
                                             pre_embedding_path=pre_w2v).init_embedding()
        self.words_embedding.weight.requires_grad=False
        self.pos_embedding = AcnnEmbedding(56, 50).init_embedding()
        # 关系嵌入
        self.W_L = Variable(torch.randn(size=(2, 10, 100)))
        # attention first

        # res = A_matrix_construct(head_matrix, end_matrix)
        # cnn
        # 输出16通道，kenel:3 * 3
        self.cnn = acnn_layer(1, 16, (3, 150), (1, 150))
        self.Bias_L = Variable(torch.randn(size=(2, 16, 3)))
        # pool attention
        self.U = Variable(torch.randn(size=(2, 16, 10)))

    def _A_matrix_construct(self, startpos_list):
        startpos_list_embedding = self.words_embedding(startpos_list)
        startpos_list_embedding_rsp = startpos_list_embedding.repeat(1, 3).reshape(2, -1, 50)
        return startpos_list_embedding_rsp

    def _R_matrix_construct(self, start, end, x_embding):
        R = []
        for i in range(start.shape[0]):
            start_i = start[i]
            end_i = end[i]
            R_i = A_matrix_construct(start_i, end_i)
            R.append(R_i.tolist())
        return torch.mul(x_embding, torch.from_numpy(np.array(R, dtype='float32')).unsqueeze(-1))

    def G_uniform(self, g_matrix):
        g_matrix = torch.exp(g_matrix)
        fenmu = g_matrix.sum(2).reshape(2, 3, 1)
        g_matrix = g_matrix / fenmu.repeat(1, 1, 100)
        return g_matrix

    def forward(self, sent_x, pos_left, pos_right, head_startpos_list, end_startpos_list, y):
        word_embedding = self.words_embedding(sent_x)
        pos_left_embedding = self.pos_embedding(pos_left)
        pos_right_embedding = self.pos_embedding(pos_right)

        # 构建A矩阵
        start_e_A = self._A_matrix_construct(head_startpos_list).bmm(word_embedding.reshape(2, 50, 3))
        end_e_A = self._A_matrix_construct(end_startpos_list).bmm(word_embedding.reshape(2, 50, 3))
        word_embedding = (start_e_A.mul(end_e_A)).bmm(word_embedding)
        net_embedding = torch.cat((word_embedding, pos_left_embedding, pos_right_embedding), 2)
        R = self._R_matrix_construct(start_e_A, end_e_A, net_embedding)
        # CNN输出结果
        cnn_result = self.cnn(R.unsqueeze(1))
        # 获得cnn compose 矩阵 R

        R_star = torch.tanh(cnn_result)

        G = R_star.permute(0, 2, 1).bmm(self.U).bmm(self.W_L)  # W^L relation embedding
        G = self.G_uniform(G)
        W_O = R_star.bmm(G)
        return W_O

if __name__ == '__main__':
    pos_embedding_id_int = dict(enumerate(np.arange(-28, 28, 1)))
    pos_embedding_id_int.update({56: 999})
    pos_embedding_int_id = {j: i for i, j in pos_embedding_id_int.items()}

    M = Acnn()
    res = M.forward(torch.from_numpy(np.array([[1,2,3], [2,1,3]])).long(),
                    torch.from_numpy(np.array([[1, 2, 0], [2, 2, 2]])).long(),
                    torch.from_numpy(np.array([[2, 2, 0], [2, 2, 2]])).long(),
                    torch.from_numpy(np.array([0, 1])).long(),
                    torch.from_numpy(np.array([0, 1])).long(),
                    torch.from_numpy(np.array([1, 2])).long())
    print(res.shape)
    import torch.optim as optim
    model = Acnn()
    optimer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
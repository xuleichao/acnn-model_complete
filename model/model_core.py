import torch
import numpy as np
import torch.nn as nn
from model.embedding import AcnnEmbedding
from model.window_slide_utils import window_slide
from attention.first_attention import diag_array_init, A_matrix_construct

pre_w2v = '/home/codes/pytorch-acnn-model/attention/embeddings.txt'

class Acnn(nn.Module):

    def __init__(self):
        super(Acnn, self).__init__()
        # embending
        self.words_embedding = AcnnEmbedding(13000, 50,
                                             pre_embedding_path=pre_w2v).init_embedding()
        self.pos_embedding = AcnnEmbedding(56, 50).init_embedding()

        # window_slide(9, 1, 8)
        # attention
        # cnn
        # pool attention

    def forward(self, sent_x, pos_left, pos_right, y):
        word_embedding = self.words_embedding(sent_x)
        pos_left_embedding = self.pos_embedding(pos_left)
        pos_right_embedding = self.pos_embedding(pos_right)
        net_embending = torch.cat((word_embedding, pos_left_embedding, pos_right_embedding), 1)
        print(net_embending)
        print(net_embending.shape)

if __name__ == '__main__':
    pos_embedding_id_int = dict(enumerate(np.arange(-28, 28, 1)))
    pos_embedding_id_int.update({56: 999})
    pos_embedding_int_id = {j: i for i, j in pos_embedding_id_int.items()}

    M = Acnn()
    res = M.forward(torch.from_numpy(np.array([[1,2,3], [2,1,3]])),
                    torch.from_numpy(np.array([[1], [2]])),
                    torch.from_numpy(np.array([[2], [2]])),
                    torch.from_numpy(np.array([1,2])))

import torch
import torch.nn as nn
from model.embeding import AcnnEmbedding
from model.window_slide_utils import window_slide
from attention.first_attention import diag_array_init, A_matrix_construct

class Acnn(nn.Module):

    def __init__(self):
        # embending
        self.words_attention = AcnnEmbedding(13000, 50, pre_embedding_path=None)
        self.pos_attention = AcnnEmbedding(28, 50)
        # window_slide(9, 1, 8)
        # attention
        # cnn
        # pool attention

    def forward(self, sent_x, e1_start, e2_start, y):
        pass


if __name__ == '__main__':
    M = Acnn()

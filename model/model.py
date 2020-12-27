import torch
import torch.nn as nn
from model.embeding import AcnnEmbedding
from model.window_slide_utils import window_slide
from attention.first_attention import diag_array_init, A_matrix_construct

class Acnn(nn.Module):

    def __init__(self):
        # embending
        # attention
        # cnn
        # pool attention
        pass

    def forward(self):
        pass


if __name__ == '__main__':
    M = Acnn()

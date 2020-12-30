import torch
import torch.nn as nn


def acnn_layer(in_channels, out_channels, kernel_size, stride=1):
    acnn = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1)
    return acnn

class Acnn:

    def __init__(self):
        pass

if __name__ == '__main__':

    pass
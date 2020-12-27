import torch
import torch.nn as nn
import numpy as np

def diag_array_init(shape_0, shape_1):
    matrix = torch.randn(shape_0, shape_1)
    return matrix

def attention_compose_plus_define(matrix_1, matrix_2):
    matrix = (matrix_1 + matrix_2) / 2
    return torch.diag(matrix)

def A_matrix_construct(matrix_head, matrix_end):
    head_trace_matrix = torch.trace(matrix_head)
    end_trace_matrix = torch.trace(matrix_end)

    matrix_head_new = torch.exp(matrix_head) / torch.exp(head_trace_matrix)
    matrix_end_new = torch.exp(matrix_end) / torch.exp(end_trace_matrix)

    return attention_compose_plus_define(matrix_head_new, matrix_end_new)


if __name__ == '__main__':
    matrix1 = diag_array_init(10, 10)
    matrix2 = diag_array_init(10, 10)

    res = A_matrix_construct(matrix1, matrix2)
    print(res)
import numpy as np

def slide_utils(this_index, head_start, end_start):
    head_relative_pos = head_start - this_index
    end_relative_pos = end_start - this_index
    return head_relative_pos, end_relative_pos

def window_slide(sentence_length, head_start, end_start):
    '''窗口操作，产生相对位置向量'''
    init_matrix = np.arange(sentence_length)
    pos_left = head_start - init_matrix
    pos_right = end_start - init_matrix
    return pos_left, pos_right

if __name__ == '__main__':
    res = window_slide(9, 1, 8)
    print(res)

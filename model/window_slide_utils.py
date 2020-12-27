
def slide_utils(this_index, head_start, end_start):
    head_relative_pos = head_start - this_index
    end_relative_pos = end_start - this_index
    return head_relative_pos, end_relative_pos

def window_slide(sentence_length, head_start, end_start):
    result = []
    for i in range(sentence_length):
        res = slide_utils(i, head_start, end_start)
        result.append(res)
    return result

if __name__ == '__main__':
    res = window_slide(9, 1, 8)
    print(res)

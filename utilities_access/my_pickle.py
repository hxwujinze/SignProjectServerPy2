# coding:utf-8
import numpy as np

def loads(string):
    rows = string.split(';')
    data_mat = []
    for each_row in rows:
        if each_row == '':
            continue
        row_data = []
        for each_col in each_row.split(','):
            if each_col == '':
                continue
            row_data.append(float(each_col))
        data_mat.append(row_data)

    return np.array(data_mat)

def dumps(array):
    """
    将任意2维np array转为string
    :param array: np.array
    :return:  string
    """
    res = ''
    try:
        dim1_len = len(array)
        dim2_len = len(array[0, :])
    except IndexError:
        print('not 2d array')
        return ''
    for row in range(dim1_len):
        for col in range(dim2_len):
            res += str(array[row][col])
            res += ','
        res += ';'
    return res

if __name__ == '__main__':
    a = np.array([np.arange(0, 200, 1) for each in range(50)])
    s = dumps(a)
    print(s)
    aa = loads(s)
    print(aa)

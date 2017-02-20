'''
# =============================================================================
#      FileName: test.py
#          Desc: 
#        Author: ZaneQi
#         Email: qizheng1993hit@gmail.com
#      HomePage: https://github.com/chouxi
#       Version: 0.0.1
#    LastChange: 2017-02-20 16:10:10
#       History:
# =============================================================================
'''
from scipy.spatial.distance import cdist

import numpy as np
from proc_files import read_test_files
from train import normalize
from train import index_2_char

def test_features(mean, std, file_name):
    features_list = []
    centers_list = []
    res = read_test_files(file_name)
    features_list = res[0]
    normalize(features_list, mean, std)
    return [features_list, res[1]]

def k_nn(D_index, char_index, k):
    prediction = [{}]
    for j in range(k):
        for i in range(len(D_index[:,j])):
            cur_char = index_2_char(D_index[i][j],char_index)
            if len(prediction) == i:
                prediction.append({})
                prediction[i].setdefault(cur_char, 1)
            else:
                if prediction[i].has_key(cur_char):
                    prediction[i][cur_char] += 1
                else:
                    prediction[i].setdefault(cur_char, 1)
    result = []
    for d in prediction:
        tmp = max(d.items(), key=lambda x: x[1])
        if tmp[1] == 1:
            result.append(d.keys()[0])
        else:
            result.append(tmp[0])
    return result
def recognition(test_features, data_base, char_index, k=5):
    D = cdist(test_features, data_base)
    D_index = np.argsort(D, axis=1)
    return k_nn(D_index, char_index, k)

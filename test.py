'''
# =============================================================================
#      FileName: test.py
#          Desc: 
#        Author: ZaneQi
#         Email: qizheng1993hit@gmail.com
#      HomePage: https://github.com/chouxi
#       Version: 0.0.1
#    LastChange: 2017-02-15 22:57:33
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
    features_list = read_test_files(file_name)
    normalize(features_list, mean, std)
    return features_list

def recognition(test_features, data_base, char_index):
    D = cdist(test_features, data_base)
    print len(D)
    D_index = np.argsort(D, axis=1)
    prediction = []
    for i in range(len(D_index[:,0])):
        prediction.append(index_2_char(D_index[i][0],char_index))
    return prediction

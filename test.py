"""
@Author:	Zane Qi
@E-mail:	qizheng1993hit@gamil.com
@Date:		12/Feb/2017 (Sun) 18:52:42
"""

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
    D_index = np.argsort(D, axis=1)
    prediction = []
    for i in range(len(D_index[:,0])):
        prediction.append(index_2_char(D_index[i][0],char_index))
    return prediction
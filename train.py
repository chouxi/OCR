'''
# =============================================================================
#      FileName: train.py
#          Desc: 
#        Author: ZaneQi
#         Email: qizheng1993hit@gmail.com
#      HomePage: https://github.com/chouxi
#       Version: 0.0.1
#    LastChange: 2017-02-20 13:16:32
#       History:
# =============================================================================
'''
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import math

from proc_files import read_files
def calc_mean(feature_list):
    mean = []
    size = len(feature_list[0])
    for i in range(size):
        tmp_sum = 0.0
        for feature in feature_list:
            tmp_sum += feature[i]
        mean.append(tmp_sum / len(feature_list))
    return mean
    
def calc_std(feature_list, mean):
    std = []
    size = len(feature_list[0])
    for i in range(size):
        tmp_sum = 0.0
        for feature in feature_list:
            tmp_sum += (feature[i] - mean[i]) * (feature[i] - mean[i])
        std.append(math.sqrt(tmp_sum / len(feature_list)))
    return std

def normalize(features_list, mean, std):
    row = len(features_list)
    for i in range(row):
        ft = len(features_list[i])
        for j in range(ft):
            features_list[i][j] = (features_list[i][j] - mean[j]) / std[j]

def index_2_char(index, char_index):
    result = ''
    for i in range(len(char_index)):
        if index < char_index[i][1]:
            result = char_index[i][0]
            break
    return result

def train_features(file_list, file_path, post_fix, enhancement_hash, enhancement_flag):
    features_list = []
    char_index = []
    for f_name in file_list:
        features_list += read_files(file_path, f_name, post_fix, enhancement_hash, enhancement_flag)
        char_index.append((f_name, len(features_list)))
    mean = calc_mean(features_list)
    std = calc_std(features_list, mean)
    #mean = np.mean(features_list)
    #std = np.std(features_list)
    normalize(features_list, mean, std)
    D = cdist(features_list, features_list)
    #io.imshow(D)
    #plt.title('Dis Mat')
    #io.show()
    D_index =np.argsort(D, axis=1)
    Ytrue = []
    Ypred = []
    for r in range(len(D_index)):
        Ytrue.append(index_2_char(r, char_index))
        Ypred.append(index_2_char(D_index[r][1], char_index))
    conf_mat = confusion_matrix(Ytrue, Ypred)
    # io.imshow(conf_mat)
    # plt.title('Confusion Matrix')
    # io.show()
    return [mean, std, features_list, char_index]

'''
# =============================================================================
#      FileName: train.py
#          Desc: 
#        Author: ZaneQi
#         Email: qizheng1993hit@gmail.com
#      HomePage: https://github.com/chouxi
#       Version: 0.0.1
#    LastChange: 2017-02-15 22:57:49
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
def normalize(features_list, mean, std):
    row = len(features_list)
    for i in range(row):
        ft = len(features_list[i])
        for j in range(ft):
            features_list[i][j] = (features_list[i][j] - mean) / std

def index_2_char(index, char_index):
    result = ''
    for i in range(len(char_index)):
        if index < char_index[i][1]:
            result = char_index[i][0]
            break
    return result

def train_features():
    file_list = ['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z']
    file_path = './H1-16images/'
    post_fix = '.bmp'
    features_list = []
    char_index = []
    for f_name in file_list:
        features_list += read_files(file_path, f_name, post_fix)
        char_index.append((f_name, len(features_list)))
    #mean = calc_mean(features_list)
    #std = calc_std(features_list, mean)
    mean = np.mean(features_list)
    std = np.std(features_list)
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

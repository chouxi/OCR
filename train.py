"""
@Author:    Zane Qi
@E-mail:    qizheng1993hit@gamil.com
@Date:      11/Feb/2017 (Sat) 19:17:00
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from scipy.spatial.distance import cdist
import math

def compute_hu(roi):
    m = moments(roi)
    cr = m[0, 1] / m[0, 0]
    cc = m[1, 0] / m[0, 0]
    mu = moments_central(roi, cr, cc)
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    return hu

def binirize(threshold, img):
    img_binary = (img < threshold).astype(np.double)
    img_label = label(img_binary, background=0)
    # print np.amax(img_label)
    regions = regionprops(img_label)
    # io.imshow(img_binary)
    ax = plt.gca()
    sum_size = 0
    count = 0
    Features=[]
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        sum_size += (maxc-minc) * (maxr-minr)
        count += 1
    # Set the 1/5 of the mean as the size_threshould
    # Because there need different size_threshould for each character.
    size_thre = sum_size / (count * 5)
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if (maxc-minc) * (maxr-minr) < size_thre:
            continue
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        hu = compute_hu(img_binary[minr:maxr, minc:maxc])
        Features.append(hu)
    # ax.title('Bounding Boxes')
    # io.show()
    return Features

def read_files(path, file_name, post_fix, show_pic=False):
    img = io.imread(path + file_name + post_fix)
    #print img.shape
    # io.imshow(img)
    # plt.title('A img')
    # io.show()
    # hist = exposure.histogram(img)
    # plt.bar(hist[1], hist[0])
    # plt.title('Histogram')
    # plt.show()
    return binirize(200, img)

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

def collecting_features():
    file_list = ['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z']
    file_path = './H1-16images/'
    post_fix = '.bmp'
    features_list = []
    char_index = []
    for f_name in file_list:
        features_list += read_files(file_path, f_name, post_fix)
        char_index.append((f_name, len(features_list)))
    # mean = calc_mean(features_list)
    # std = calc_std(features_list, mean)
    mean = np.mean(features_list)
    std = np.std(features_list)
    normalize(features_list, mean, std)
    D = cdist(features_list, features_list)
    # io.imshow(D)
    # plt.title('Dis Mat')
    # io.show()
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

if __name__ == '__main__':
    collecting_features()

'''
# =============================================================================
#      FileName: proc_files.py
#          Desc: 
#        Author: ZaneQi
#         Email: qizheng1993hit@gmail.com
#      HomePage: https://github.com/chouxi
#       Version: 0.0.1
#    LastChange: 2017-02-20 12:50:29
#       History:
# =============================================================================
'''

import numpy as np
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def dilation_erosion(origin_bi_img):
    #dilation_mat = [[0,1,1,0],[1,1,1,1],[0,1,1,0],[0,1,1,0]]
    dilation_mat = np.ones((3,3))
    dilation_img = morphology.binary_dilation(origin_bi_img, selem=dilation_mat).astype(np.double)
    erosion_mat = [[0,1,0],[1,1,1],[0,1,0]]
    img_binary = morphology.binary_erosion(dilation_img, selem=erosion_mat).astype(np.double)
    return img_binary

def compute_hu(roi):
    m = moments(roi)
    cr = m[0, 1] / m[0, 0]
    cc = m[1, 0] / m[0, 0]
    mu = moments_central(roi, cr, cc)
    nu = moments_normalized(mu)
    hu = moments_hu(nu)
    return hu

def binirize(threshold, img):
    threshould_img = (img < threshold).astype(np.double)
    img_binary = dilation_erosion(threshould_img)
    img_label = label(img_binary, background=0)
    # print np.amax(img_label)
    regions = regionprops(img_label)
    #io.imshow(img_binary)
    ax = plt.gca()
    Features=[]
    centers=[]
    sum_length = 0.0
    sum_width = 0.0
    count = len(regions)
    ratio = 3.0
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        sum_length += (maxc-minc)
        sum_width += (maxr-minr)
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if (maxc-minc) < sum_length / (count * ratio) or (maxr-minr) < sum_width / (count * ratio) or (maxc-minc) > sum_length / count * ratio or (maxr-minr) > sum_width / count * ratio:
            continue
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        hu = compute_hu(img_binary[minr:maxr, minc:maxc])
        Features.append(hu)
        centers.append(props.centroid)
    print len(Features)
    #io.show()
    return [Features, centers]

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
    return binirize(180, img)[0]

def read_test_files(file_name, show_pic=False):
    img = io.imread(file_name)
    return binirize(180, img)

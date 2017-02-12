"""
@Author:    Zane Qi
@E-mail:    qizheng1993hit@gamil.com
@Date:      11/Feb/2017 (Sat) 19:17:00
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle

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
    print np.amax(img_label)
    regions = regionprops(img_label)
    io.imshow(img_binary)
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

def read_files():
    img = io.imread('./H1-16images/z.bmp')
    print img.shape
    # io.imshow(img)
    # plt.title('A img')
    # io.show()

    # hist = exposure.histogram(img)
    # plt.bar(hist[1], hist[0])
    # plt.title('Histogram')
    # plt.show()
    binirize(200, img)
if __name__ == '__main__':
    read_files()
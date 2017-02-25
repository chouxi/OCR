'''
# =============================================================================
#      FileName: proc_files.py
#          Desc: 
#        Author: ZaneQi
#         Email: qizheng1993hit@gmail.com
#      HomePage: https://github.com/chouxi
#       Version: 0.0.1
#    LastChange: 2017-02-24 22:52:49
#       History:
# =============================================================================
'''

import numpy as np
from skimage.measure import label, regionprops, moments, moments_central, moments_normalized, moments_hu, perimeter
from skimage.filters import thresholding
from skimage import io, exposure, morphology
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import math

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
    hu = moments_hu(nu).tolist()
    return hu

def binirize(threshold, img, enhancement_hash, enhancement_flag, add_text=False, prediction=[]):
    i = 0
    if enhancement_flag & enhancement_hash['morphology']:
        #print 'morphology'
        threshould_img = (img < threshold).astype(np.double)
        img_binary = dilation_erosion(threshould_img)
    else:
        img_binary = (img < threshold).astype(np.double)
    img_label = label(img_binary, background=0)
    # print np.amax(img_label)
    regions = regionprops(img_label)
    io.imshow(img_binary)
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
    lower_c = sum_length / (count * ratio)
    lower_r = sum_width / (count * ratio)
    upper_c = sum_length / count * ratio
    upper_r = sum_width / count * ratio
    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if enhancement_flag & enhancement_hash['size_threshold']:
            #print 'size_threshold'
            if (maxc-minc) < lower_c or (maxr-minr) < lower_r or (maxc-minc) > upper_c or (maxr-minr) > upper_r:
                continue
        else:
            if (maxc-minc) < 10 or (maxr-minr) < 10:
                continue
        #if (maxc-minc) < 10 or (maxr-minr) < 10 or (maxc-minc) > 100 or (maxr-minr) > 100:
        hu = compute_hu(img_binary[minr:maxr, minc:maxc])
        pr = perimeter(img_binary[minr:maxr, minc:maxc])
        area = (maxc-minc) * (maxr-minr)
        if enhancement_flag & enhancement_hash['circularity']:
            #print 'circularity'
            #circularity
            hu.append((area / (pr*pr)) * 4 * math.pi)
        if enhancement_flag & enhancement_hash['density']:
            #print 'density'
            #density
            hu.append(area /float(props.convex_area))
        if enhancement_flag & enhancement_hash['convexity']:
            #print 'convexity'
            #convexity
            convex_img = props.convex_image
            pr_conv = perimeter(convex_img)
            hu.append(pr_conv - pr)
        Features.append(hu)
        centers.append(props.centroid)
        if add_text:
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
            plt.text(props.centroid[1] + 20,props.centroid[0] +20, prediction[i],color='green', fontsize=20)
            i += 1
    plt.title("bounding box")
    #print len(Features)
    if add_text:
        io.show()
        #print threshold
        #print lower_c, lower_r, upper_c, upper_r
    return [Features, centers]

def read_files(path, file_name, post_fix, enhancement_hash, enhancement_flag, add_text=False, prediction=[]):
    img = io.imread(path + file_name + post_fix)
    #print img.shape
    # io.imshow(img)
    # plt.title('A img')
    # io.show()
    # hist = exposure.histogram(img)
    # plt.bar(hist[1], hist[0])
    # plt.title('Histogram')
    # plt.show()
    if enhancement_flag & enhancement_hash['image_threshold']:
        return binirize(thresholding.threshold_yen(img), img, enhancement_hash, enhancement_flag, add_text, prediction)[0]
    else:
        return binirize(200, img, enhancement_hash, enhancement_flag, add_text, prediction)[0]

def read_test_files(file_name, enhancement_hash, enhancement_flag, add_text=False, prediction=[]):
    img = io.imread(file_name)
    if enhancement_flag & enhancement_hash['image_threshold']:
        #print 'image_threshold'
        return binirize(thresholding.threshold_yen(img), img, enhancement_hash, enhancement_flag, add_text, prediction)
    else:
        return binirize(200, img, enhancement_hash, enhancement_flag, add_text, prediction)

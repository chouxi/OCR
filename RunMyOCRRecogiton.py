'''
# =============================================================================
#      FileName: RunMyOCRRecogiton.py
#          Desc: 
#        Author: ZaneQi
#         Email: qizheng1993hit@gmail.com
#      HomePage: https://github.com/chouxi
#       Version: 0.0.1
#    LastChange: 2017-02-23 13:24:56
#       History:
# =============================================================================
'''
import test
import train
import pickle
import argparse

def get_ratio_recg(feature_list, true_char):
    add = 1.0
    for feature in feature_list:
        if feature == true_char:
            add += 1
    return add / len(feature_list)
            
def get_ratio(true_classes, true_locations, test_classes, test_locations):
    diff = 10
    right_count = 0.0
    for location, feature in zip(test_locations, test_classes):
        for t_loc, t_feature in zip(true_locations, true_classes):
            if abs(t_loc[1] - location[0]) < diff and abs(t_loc[0] - location[1]) < diff and t_feature == feature:
                right_count += 1
                break;
    return right_count / len(true_classes)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="input test file and ground truth file")
    parser.add_argument('-test_file', help='test file name', required = True)
    parser.add_argument('-ground_truth', help='ground_truth file name', required = True)
    args = parser.parse_args()
    file_list = ['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z']
    file_path = './H1-16images/'
    post_fix = '.bmp'
    train_data = train.train_features(file_list, file_path, post_fix)
    for char in file_list:
        test_data = test.test_features(train_data[0], train_data[1], file_path+char+post_fix)
        tmp = []
        test_classes = test.recognition(test_data[0], train_data[2],train_data[3], 5)
        print get_ratio_recg(test_classes[0], char)
        train.read_files(file_path, char, post_fix, True, test_classes[0])
    test_data = test.test_features(train_data[0], train_data[1], args.test_file)
    test_classes = test.recognition(test_data[0], train_data[2],train_data[3], 11)
    #print len(test_classes[0])

    # Must update the title before show
    train.plt.title("distance matrix")
    train.io.imshow(test_classes[1])
    train.io.show()

    pkl_file = open(args.ground_truth, 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    print get_ratio(mydict['classes'], mydict['locations'], test_classes[0], test_data[1])
    # just show plt with text
    test.read_test_files(args.test_file, True, test_classes[0])
    '''
    for i in range(2,20):
        test_classes = test.recognition(test_data[0], train_data[2],train_data[3], i)
        pkl_file = open(args.ground_truth, 'rb')
        mydict = pickle.load(pkl_file)
        pkl_file.close()
        print i, get_ratio(mydict['classes'], mydict['locations'], test_classes[0], test_data[1])
    '''

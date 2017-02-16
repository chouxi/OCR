'''
# =============================================================================
#      FileName: RunMyOCRRecogiton.py
#          Desc: 
#        Author: ZaneQi
#         Email: qizheng1993hit@gmail.com
#      HomePage: https://github.com/chouxi
#       Version: 0.0.1
#    LastChange: 2017-02-15 22:57:20
#       History:
# =============================================================================
'''
import test
from train import train_features

if __name__ == '__main__':
    train_data = train_features()
    test_data = test.test_features(train_data[0], train_data[1], './H1-16images/test2.bmp')
    result = test.recognition(test_data, train_data[2],train_data[3])
    print result

"""
@Author:	Zane Qi
@E-mail:	qizheng1993hit@gamil.com
@Date:		12/Feb/2017 (Sun) 19:09:43
"""

import test
from train import train_features

if __name__ == '__main__':
    train_data = train_features()
    test_data = test.test_features(train_data[0], train_data[1], './H1-16images/test1.bmp')
    result = test.recognition(test_data, train_data[2],train_data[3])
    print result
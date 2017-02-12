"""
@Author:	Zane Qi
@E-mail:	qizheng1993hit@gamil.com
@Date:		12/Feb/2017 (Sun) 11:31:33
"""
from train import read_files

def collecting_features():
	file_list = ['a', 'd', 'f', 'h', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'u', 'w', 'x', 'z']
	file_path = './H1-16images/'
	post_fix = '.bmp'
	features_dict = {}
	for f_name in file_list:
		features_dict[f_name] = read_files(file_path, f_name, post_fix)

	for (key, value) in features_dict.items():
		for val in value:
			print len(value)

if __name__ == '__main__':
	collecting_features()
import sys
import numpy as np
import argparse
import csv
import keras
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
import pickle

TOP_WORD=1000
MAX_LINE=50
VECTOR_LENGTH=32

def readfile(test_filepath):
	dict = {}
	with open('filename.pickle', 'rb') as handle:
		dict = pickle.load(handle)
	testfile = open(test_filepath,'r',encoding='utf-8')
	test = testfile.readlines()
	count = len(dict)
	for index, row in enumerate(test):
		if index != 0:
			test[index] = row.split(',',1)[1]
			test[index] = test[index].split( )
			for i, e in enumerate(test[index]):
				if dict.get(e, -1) != -1:
					test[index][i] = dict[e]
				else:
					dict.update({e:count})
					test[index][i] = dict[e]
					count = count+1
	test = test[1:]
	print(test[0])
	return test
def main(opts):
	x_test = readfile(opts.test_data_path)
	x_test = sequence.pad_sequences(x_test, maxlen=MAX_LINE)
	submission = open(opts.sub_data_path,'w')
	print("Testing size:"+str(x_test.shape[0]))
	model = load_model(opts.model_data_path)
	result = model.predict(x_test, batch_size = 64, verbose = 1)
	writer = csv.writer(submission)
	writer.writerow(["id","label"])
	for index,element in enumerate(result):
		if element>0.5:
			writer.writerow([index,1])
		else:
			writer.writerow([index,0])
	print("Done")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data', type=str,default='/home/guest/wariard/ylc/ylc/hw4/data/training_label.txt',dest='train_data_path',help='Training data path')
	parser.add_argument('--test_data', type=str,default='/home/guest/wariard/ylc/ylc/hw4/data/testing_data.txt',dest='test_data_path',help='Testing data path')
	parser.add_argument('--sub_data',type=str,default='submission.csv',dest='sub_data_path',help='Sub semi')
	parser.add_argument('--model_data',type=str,default='model.h5',dest='model_data_path',help='model name')
	parser.add_argument('--model_out',type=str,default='model.h5',dest='model_out_path',help='model name')
	parser.add_argument('--emsemble',type=int,default='1',dest='emsemble',help='model name')
	parser.add_argument('-semi',action='store_true', default=False,dest='semi',help='Do semi supervise')
	parser.add_argument('-train',action='store_true', default=False,dest='train',help='Do train')
	parser.add_argument('-test',action='store_true', default=False,dest='test',help='Do test')
	parser.add_argument('-emsem',action='store_true', default=False,dest='emsem',help='Do emsemble')
	opts = parser.parse_args()
	main(opts)
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

#/home/guest/wariard/ylc/ylc/hw4/data

TOP_WORD=1000
MAX_LINE=50
VECTOR_LENGTH=32

def get_session():
	gpu_opt = tf.GPUOptions(allow_growth=True)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt, allow_soft_placement=True))

def readfile(filepath, test_filepath,opts):
	global TOP_WORD
	datafile = open(filepath,'r',encoding='utf-8')
	testfile = open(test_filepath,'r',encoding='utf-8')
	data = datafile.readlines()
	label = []
	test = testfile.readlines()
	dict = {}
	count=0

	for index, row in enumerate(data):
		data[index] = data[index].split( )
		label.append(data[index][0])
		data[index] = data[index][2:]

		for i, e in enumerate(data[index]):
			if dict.get(e, -1) != -1:
				data[index][i] = dict[e]
			else:
				dict.update({e:count})
				data[index][i] = dict[e]
				count = count+1
	if opts.test:
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
	elif opts.semi:
		for index, row in enumerate(test):
			if index != 0:
				#test[index] = row.split(',',1)[1]
				test[index] = test[index].split( )
				for i, e in enumerate(test[index]):
					if dict.get(e, -1) != -1:
						test[index][i] = dict[e]
					else:
						dict.update({e:count})
						test[index][i] = dict[e]
						count = count+1
		test = test[1:]
	TOP_WORD = count
	return data, label, test

def main(opts):
	ktf.set_session=(get_session())

	model_name=opts.model_data_path 
	x_train, y_train, x_test = readfile(opts.train_data_path, opts.test_data_path,opts)
	x_train = sequence.pad_sequences(x_train, maxlen=MAX_LINE)
	print("Training size:"+str(x_train.shape[0]))

	#train
	if opts.train:
		print("TOP WORD:"+str(TOP_WORD))
		model = Sequential()
		model.add(Embedding(TOP_WORD, VECTOR_LENGTH, input_length=MAX_LINE))
		model.add(LSTM(100))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(model.summary())
		callbacks = keras.callbacks.ModelCheckpoint(opts.model_out_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
		model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_split=0.1,callbacks=[callbacks])
	
	#predict
	if opts.test:
		x_test = sequence.pad_sequences(x_test, maxlen=MAX_LINE)
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
	if opts.emsem:
		print(opts.emsemble/2)
		result_sum = np.zeros((200000,1))
		submission = open('submission.csv','w')
		for i in range(1,opts.emsemble+1):
			print('model:'+str(i))
			model = load_model('model'+str(i)+'.h5')
			result = model.predict(x_test, batch_size = 64, verbose = 1)
			result = np.array(result,dtype = float)
			for index,element in enumerate(result):
				if element>0.5:
					result_sum[index] = result_sum[index]+1

		writer = csv.writer(submission)
		writer.writerow(["id","label"])
		for index,element in enumerate(result_sum):
			if element >= 2.5:
				writer.writerow([index,1])
			else:
				writer.writerow([index,0])

	if opts.semi:
		x_test = sequence.pad_sequences(x_test, maxlen=MAX_LINE)
		print("Testing size:"+str(x_test.shape[0]))
		model = load_model(model_name)
		result = model.predict(x_test, batch_size = 64, verbose = 1)
		semi = []
		semi_label = []
		for index,element in enumerate(result):
			if element>0.8:
				semi.append(x_test[index])
				semi_label.append(1)
			elif element<0.2:
				semi.append(x_test[index])
				semi_label.append(0)
		x_train = np.append(x_train,semi,axis=0)
		y_train.extend(semi_label)
		x_val = x_train[:10000]
		y_val = y_train[:10000]
		x_train = x_train[10000:]
		y_train = y_train[10000:]
		print(" New Training size:"+str(x_train.shape[0]))
		model = Sequential()
		model.add(Embedding(TOP_WORD, VECTOR_LENGTH, input_length=MAX_LINE))
		model.add(LSTM(100))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(model.summary())
		callbacks = keras.callbacks.ModelCheckpoint(opts.model_out_path, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
		model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_val,y_val),callbacks=[callbacks])

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data', type=str,default='/home/guest/wariard/ylc/ylc/hw4/data/training_label.txt',dest='train_data_path',help='Training data path')
	parser.add_argument('--test_data', type=str,default='/home/guest/wariard/ylc/ylc/hw4/data/testing_data.txt',dest='test_data_path',help='Testing data path')
	parser.add_argument('--semi_data',type=str,default='/home/guest/wariard/ylc/ylc/hw4/data/training_nolabel.txt',dest='train_semi',help='Train semi')
	parser.add_argument('--model_data',type=str,default='model.h5',dest='model_data_path',help='model name')
	parser.add_argument('--model_out',type=str,default='model.h5',dest='model_out_path',help='model name')
	parser.add_argument('--emsemble',type=int,default='1',dest='emsemble',help='model name')
	parser.add_argument('-semi',action='store_true', default=False,dest='semi',help='Do semi supervise')
	parser.add_argument('-train',action='store_true', default=False,dest='train',help='Do train')
	parser.add_argument('-test',action='store_true', default=False,dest='test',help='Do test')
	parser.add_argument('-emsem',action='store_true', default=False,dest='emsem',help='Do emsemble')

	opts = parser.parse_args()
	main(opts)
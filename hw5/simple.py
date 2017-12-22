from keras.models import Model
from keras.layers import Flatten, Input, Dense, Dropout
from keras.layers import Embedding
import keras
from keras.models import load_model
import argparse
import sys
import numpy as np
import csv
import keras.backend as K

USER_COUNT = 0
MOVIE_COUNT = 0

def readfile(datapath):
	global USER_COUNT
	global MOVIE_COUNT
	with open(datapath) as f:
		reader = csv.reader(f)
		data = list(reader)

	data = np.array(data[1:], dtype=float)
	np.random.seed(3318)
	index = np.random.permutation(len(data))
	data = data[index]
	user = data[:, 1]
	movie = data[:, 2]
	rating = data[:, 3]
	USER_COUNT = int(np.max(user)) + 1
	MOVIE_COUNT = int(np.max(movie)) + 1
	return (user, movie, rating)

def readtest(datapath):
	with open(datapath) as f:
		reader = csv.reader(f)
		data = list(reader)
	data = np.array(data[1:], dtype=float)
	user = data[:, 1]
	movie = data[:, 2]
	return (user, movie)

def rmse(y_true, y_pred): return K.sqrt(K.mean((y_pred - y_true)**2))

def main(opts):
	'''x_train_user, x_train_movie, y_train = readfile(opts.train_data_path)
	print('Movie size:'+str(MOVIE_COUNT))
	print('User size:'+str(USER_COUNT))
	user_input = Input(shape=[1],dtype='int32',name='user_input')
	x_user = Embedding(output_dim=512,input_dim=USER_COUNT,input_length=1)(user_input)
	x_user = Flatten()(x_user)
	x_user = Dropout(0.5)(x_user)
	movie_input = Input(shape=[1],dtype='int32',name='movie_input')
	x_movie = Embedding(output_dim=512,input_dim=MOVIE_COUNT,input_length=1)(movie_input)
	x_movie = Flatten()(x_movie)
	x_movie = Dropout(0.5)(x_movie)
	x = keras.layers.dot([x_user,x_movie],axes=1)
	print(y_train)
	#x= keras.layers.Concatenate()([x_user,x_movie])
	#x=Dense(128,activation='relu')(x)
	#x=Dense(64,activation='relu')(x)
	#x=Dense(1,activation='relu')(x)
	model = Model(inputs=[user_input,movie_input],outputs=x)
	print(model.summary())

	model.compile(loss='mse', optimizer='adam', metrics=[rmse])
	callbacks = keras.callbacks.ModelCheckpoint('model1.h5', monitor='val_rmse', save_best_only=True, mode='min')
	model.fit({'user_input': x_train_user, 'movie_input': x_train_movie}, y_train, epochs=100, batch_size=9096, verbose=1, validation_split=0.1,callbacks=[callbacks])
	'''
	x_test_user, x_test_movie = readtest(opts.test_data_path)
	model_predict = load_model('model_simple.h5', custom_objects={'rmse': rmse})
	print(model_predict.summary())
	result = model_predict.predict([x_test_user,x_test_movie])
	result = np.clip(result,1,5)
	submission = open(opts.sub_data_path,'w')
	writer = csv.writer(submission)
	writer.writerow(["TestDataID","Rating"])
	for index,row in enumerate(result):
		writer.writerow([index+1,row[0]])
	print(result)

	#result = model.predict([x_train_user,x_train_movie])
	#print(result[0])

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data',type=str,default='train.csv',dest='train_data_path',help='Training data path')
	parser.add_argument('--test_data',type=str,default='test.csv',dest='test_data_path',help='Testing data path')
	parser.add_argument('--sub_data',type=str,default='sub_simple.csv',dest='sub_data_path',help='Testing data path')
	parser.add_argument('--user_data',type=str,default='test.csv',dest='user_data_path',help='Testing data path')
	parser.add_argument('--movie_data',type=str,default='test.csv',dest='movie_data_path',help='Testing data path')
	opts = parser.parse_args()
	main(opts)
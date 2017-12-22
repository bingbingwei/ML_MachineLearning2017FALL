from keras.models import Model
from keras.layers import Flatten, Input, Dense, Dropout
from keras.layers import Embedding
from keras.preprocessing import sequence
import keras
from keras.models import load_model
import argparse
import sys
import numpy as np
import csv
import keras.backend as K

np.set_printoptions(threshold=np.nan)

USER_COUNT = 6041
MOVIE_COUNT = 0
MAX_MOVIE = 3953
MAX_MOVIE_LENG = 18

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

def readmovie(datapath):
	with open(datapath,'r',encoding='utf-8') as f:
		data = f.readlines()

	x = [[] for j in range(MAX_MOVIE)]
	dict = {}
	count = 1;
	for i,row in enumerate(data):
		if i!=0:
			index  = int(row.split("::")[0])
			row = row.split("::")[2]
			row = row.rstrip()
			row = row.split('|')
			item_id = []
			for e in row:
				if dict.get(e,-1)==-1:
					dict.update({e:count})
					item_id.append(dict[e])
					count = count+1
				else:
					item_id.append(dict[e])
			x[index].extend(item_id)
	y = np.zeros((MAX_MOVIE,len(dict)),dtype =int)
	for index, row in enumerate(x):
		for ele in row:
			y[index][ele-1]=1
	return y

def readuser(datapath):
	with open(datapath,'r',encoding='utf-8') as f:
		data = f.readlines()
	y = np.zeros((USER_COUNT,3),dtype =int)
	max_occu = 0
	for i,row in enumerate(data):
		if i != 0:
			index = int(row.split("::")[0])
			element = row.split("::")[1:4]
			if element[0]=='M':
				element[0]=1
			else:
				element[0]=0
			y[index] = np.array(element,dtype=int)
			if y[index][2]>max_occu:
				max_occu = y[index][2]
	print("MAX OCCU:"+str(max_occu))
	return y

def rmse(y_true, y_pred): return K.sqrt(K.mean((y_pred - y_true)**2))

def DNN_Model():
	user_input = Input(shape=[1],dtype='int32',name='user_input')
	x_user = Embedding(output_dim=128,input_dim=USER_COUNT,input_length=1)(user_input)
	x_user = Flatten()(x_user)
	x_user = Dropout(0.5)(x_user)
	#movie train data input
	movie_input = Input(shape=[1],dtype='int32',name='movie_input')
	x_movie = Embedding(output_dim=128,input_dim=MOVIE_COUNT,input_length=1)(movie_input)
	x_movie = Flatten()(x_movie)
	x_movie = Dropout(0.5)(x_movie)
	x_concate = keras.layers.concatenate([x_user,x_movie],axis=1)
	x = Dense(128,activation='relu')(x_concate)
	x = Dense(64,activation='relu')(x)
	x = Dense(32,activation='relu')(x)
	x = Dense(1,activation='linear')(x)
	model = Model(inputs=[user_input,movie_input],outputs=x)
	print(model.summary())
	model.compile(loss='mse',optimizer='adam',metrics=[rmse])
	return model


def MF_Model():
	#user train data input
	user_input = Input(shape=[1],dtype='int32',name='user_input')
	x_user = Embedding(output_dim=128,input_dim=USER_COUNT,input_length=1)(user_input)
	x_user = Flatten()(x_user)
	x_user = Dropout(0.5)(x_user)
	x_user_bias = Embedding(output_dim=1,input_dim=USER_COUNT,input_length=1)(user_input)
	x_user_bias = Flatten()(x_user_bias)
	x_user_bias = Dropout(0.5)(x_user_bias)
	#movie train data input
	movie_input = Input(shape=[1],dtype='int32',name='movie_input')
	x_movie = Embedding(output_dim=128,input_dim=MOVIE_COUNT,input_length=1)(movie_input)
	x_movie = Flatten()(x_movie)
	x_movie = Dropout(0.5)(x_movie)
	x_movie_bias = Embedding(output_dim=1,input_dim=MOVIE_COUNT,input_length=1)(movie_input)
	x_movie_bias = Flatten()(x_movie_bias)
	x_movie_bias = Dropout(0.5)(x_movie_bias)
	x_dot = keras.layers.dot([x_user,x_movie],axes=1)
	x = keras.layers.add([x_dot, x_user_bias, x_movie_bias])
	model = Model(inputs=[user_input,movie_input],outputs=x)
	print(model.summary())
	model.compile(loss='mse', optimizer='adam', metrics=[rmse])
	return model

#add movies.csv feature
def feat_input_MF_Model():
	user_input = Input(shape=[1],dtype='int32',name='user_input')
	x_user = Embedding(output_dim=128,input_dim=USER_COUNT,input_length=1)(user_input)
	x_user = Flatten()(x_user)
	x_user = Dropout(0.5)(x_user)

	movie_input = Input(shape=[1],dtype='int32',name='movie_input')
	x_movie = Embedding(output_dim=128,input_dim=MOVIE_COUNT,input_length=1)(movie_input)
	x_movie = Flatten()(x_movie)
	x_movie = Dropout(0.5)(x_movie)
	movie_feat_input = Input(shape=[MAX_MOVIE_LENG],name='movie_feat')
	f_movie = Dense(128)(movie_feat_input)
	movie_dot = keras.layers.dot([x_movie,f_movie],axes=1)
	movie_user_dot = keras.layers.dot([x_user,x_movie],axes=1)
	user_dot = keras.layers.dot([x_user,f_movie],axes=1)
	x_concate = keras.layers.concatenate([movie_dot,movie_user_dot,user_dot],axis=1)
	x = Dense(16)(x_concate)
	x = Dense(1,activation='relu')(x)
	model = Model(inputs=[user_input,movie_input,movie_feat_input],outputs=x)
	print(model.summary())
	model.compile(loss='mse', optimizer='adam', metrics=[rmse])
	return model
#add both .csv feature
def both_feat_input_MF_Model():
	user_input = Input(shape=[1],dtype='int32',name='user_input')
	x_user = Embedding(output_dim=128,input_dim=USER_COUNT,input_length=1)(user_input)
	x_user = Flatten()(x_user)
	x_user = Dropout(0.5)(x_user)

	movie_input = Input(shape=[1],dtype='int32',name='movie_input')
	x_movie = Embedding(output_dim=128,input_dim=MOVIE_COUNT,input_length=1)(movie_input)
	x_movie = Flatten()(x_movie)
	x_movie = Dropout(0.5)(x_movie)

	movie_feat_input = Input(shape=[MAX_MOVIE_LENG],name='movie_feat')
	f_movie = Dense(128)(movie_feat_input)

	user_gender_input = Input(shape=[1],name='user_gender')
	f_user_gender = Embedding(output_dim=64,input_dim=2,input_length=1)(user_gender_input)
	f_user_gender = Flatten()(f_user_gender)
	f_user_gender = Dropout(0.5)(f_user_gender)

	user_occu_input = Input(shape=[1],name='user_occu')
	f_user_occu = Embedding(output_dim=64,input_dim=21,input_length=1)(user_occu_input)
	f_user_occu = Flatten()(f_user_occu)
	f_user_occu = Dropout(0.5)(f_user_occu)

	user_age_input = Input(shape=[1],name='user_age')

	f_user = keras.layers.concatenate([f_user_gender,f_user_occu,user_age_input],axis=1)
	f_user =Dense(128)(f_user)

	movie_fmov_dot = keras.layers.dot([x_movie,f_movie],axes=1)
	movie_fusr_dot = keras.layers.dot([x_movie,f_user],axes=1)
	movie_user_dot = keras.layers.dot([x_user,x_movie],axes=1)
	user_fusr_dot = keras.layers.dot([x_user,f_user],axes=1)
	user_fmov_dot = keras.layers.dot([x_user,f_movie],axes=1)
	fusr_fmov_dot = keras.layers.dot([f_movie,f_user],axes=1)
	x_concate = keras.layers.concatenate([movie_fmov_dot,movie_fusr_dot,movie_user_dot,user_fusr_dot,user_fmov_dot,fusr_fmov_dot],axis=1)
	x = Dense(32,activation='selu')(x_concate)
	x = Dense(1,activation='linear')(x)
	model = Model(inputs=[user_input,movie_input,movie_feat_input,user_gender_input,user_occu_input,user_age_input],outputs=x)
	print(model.summary())
	model.compile(loss='mse', optimizer='adam', metrics=[rmse])
	return model


def main(opts):
	model_name = opts.model_data_path
	if opts.trainMF:
		x_train_user, x_train_movie, y_train = readfile(opts.train_data_path)
		print('Movie size:'+str(MOVIE_COUNT))
		print('User size:'+str(USER_COUNT))
		model = MF_Model()
		callbacks = keras.callbacks.ModelCheckpoint(model_name, monitor='val_rmse', save_best_only=True, mode='min')
		model.fit({'user_input': x_train_user, 'movie_input': x_train_movie}, y_train, epochs=1000, batch_size=9096, verbose=1, validation_split=0.1,callbacks=[callbacks])
	elif opts.trainDNN:
		x_train_user, x_train_movie, y_train = readfile(opts.train_data_path)
		print('Movie size:'+str(MOVIE_COUNT))
		print('User size:'+str(USER_COUNT))
		model = DNN_Model()
		callbacks = keras.callbacks.ModelCheckpoint(model_name, monitor='val_rmse', save_best_only=True, mode='min')
		model.fit({'user_input': x_train_user, 'movie_input': x_train_movie}, y_train, epochs=1000, batch_size=9096, verbose=1, validation_split=0.1,callbacks=[callbacks])
	elif opts.predict:
		x_test_user, x_test_movie = readtest(opts.test_data_path)
		model_predict = load_model(model_name, custom_objects={'rmse': rmse})
		print(model_predict.summary())
		result = model_predict.predict([x_test_user,x_test_movie])
		result = np.clip(result,1,5)
		submission = open(opts.sub_data_path,'w')
		writer = csv.writer(submission)
		writer.writerow(["TestDataID","Rating"])
		for index,row in enumerate(result):
			writer.writerow([index+1,row[0]])
		print(result)
	elif opts.movie_feat:
		x_train_user, x_train_movie, y_train = readfile(opts.train_data_path)
		movie_feat = readmovie(opts.movie_data_path)
		x_movie_feat = []
		for row in x_train_movie:
			x_movie_feat.append(movie_feat[int(row)])
		x_movie_feat = np.array(x_movie_feat,dtype=float)
		print(x_train_movie[100])
		print(x_movie_feat[100])
		model = feat_input_MF_Model()
		callbacks = keras.callbacks.ModelCheckpoint(model_name, monitor='val_rmse', save_best_only=True, mode='min')
		model.fit({'user_input': x_train_user, 'movie_input': x_train_movie, 'movie_feat':x_movie_feat}, y_train, epochs=1000, batch_size=9096, verbose=1, validation_split=0.1,callbacks=[callbacks])
	elif opts.movie_predict:
		x_test_user, x_test_movie = readtest(opts.test_data_path)
		movie_feat = readmovie(opts.movie_data_path)
		x_movie_feat = []
		for row in x_test_movie:
			x_movie_feat.append(movie_feat[int(row)])
		x_movie_feat = np.array(x_movie_feat,dtype=float)

		model = load_model(model_name, custom_objects={'rmse': rmse})
		result = model.predict([x_test_user,x_test_movie,x_movie_feat])
		result = np.clip(result,1,5)
		submission = open(opts.sub_data_path,'w')
		writer = csv.writer(submission)
		writer.writerow(["TestDataID","Rating"])
		for index,row in enumerate(result):
			writer.writerow([index+1,row[0]])
	elif opts.train_feat:
		x_train_user, x_train_movie, y_train = readfile(opts.train_data_path)
		movie_feat = readmovie(opts.movie_data_path)
		user_feat = readuser(opts.user_data_path)
		x_movie_feat = []
		x_user_feat = []
		for row in x_train_movie:
			x_movie_feat.append(movie_feat[int(row)])
		x_movie_feat = np.array(x_movie_feat,dtype=float)	

		for row in x_train_user:
			x_user_feat.append(user_feat[int(row)])
		x_user_feat = np.array(x_user_feat,dtype=float)
		model = both_feat_input_MF_Model()
		callbacks = keras.callbacks.ModelCheckpoint(model_name, monitor='val_rmse', save_best_only=True, mode='min')
		model.fit({'user_input': x_train_user, 'movie_input': x_train_movie, 'movie_feat':x_movie_feat,'user_gender':x_user_feat[:,0],'user_occu':x_user_feat[:,2],'user_age':x_user_feat[:,1]}, y_train, epochs=1000, batch_size=9096, verbose=1, validation_split=0.1,callbacks=[callbacks])
	elif opts.feat_predict:
		x_test_user, x_test_movie = readtest(opts.test_data_path)
		movie_feat = readmovie(opts.movie_data_path)
		user_feat = readuser(opts.user_data_path)
		x_movie_feat = []
		x_user_feat = []
		for row in x_test_movie:
			x_movie_feat.append(movie_feat[int(row)])
		x_movie_feat = np.array(x_movie_feat,dtype=float)	

		for row in x_test_user:
			x_user_feat.append(user_feat[int(row)])
		x_user_feat = np.array(x_user_feat,dtype=float)
		model = load_model(model_name, custom_objects={'rmse': rmse})
		result = model.predict([x_test_user,x_test_movie,x_movie_feat,x_user_feat[:,0],x_user_feat[:,2],x_user_feat[:,1]])
		result = np.clip(result,1,5)
		submission = open(opts.sub_data_path,'w')
		writer = csv.writer(submission)
		writer.writerow(["TestDataID","Rating"])
		for index,row in enumerate(result):
			writer.writerow([index+1,row[0]])




if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data',type=str,default='train.csv',dest='train_data_path',help='Training data path')
	parser.add_argument('--movie_data',type=str,default='movies.csv',dest='movie_data_path',help='Movie data path')
	parser.add_argument('--user_data',type=str,default='users.csv',dest='user_data_path',help='User data path')
	parser.add_argument('--test_data',type=str,default='test.csv',dest='test_data_path',help='Testing data path')
	parser.add_argument('--model_data',type=str,default='model.h5',dest='model_data_path',help='Model data path')
	parser.add_argument('--sub_data',type=str,default='sub.csv',dest='sub_data_path',help='Submission data path')
	parser.add_argument('-trainMF',action='store_true', default=False,dest='trainMF',help='Do trainMF supervise')
	parser.add_argument('-trainDNN',action='store_true', default=False,dest='trainDNN',help='Do trainDNN supervise')
	parser.add_argument('-movie_feat',action='store_true', default=False,dest='movie_feat',help='Do movie feat supervise')
	parser.add_argument('-movie_predict',action='store_true', default=False,dest='movie_predict',help='Do movie predict supervise')
	parser.add_argument('-train_feat',action='store_true', default=False,dest='train_feat',help='Do train feat both movie and user supervise')	
	parser.add_argument('-feat_predict',action='store_true', default=False,dest='feat_predict',help='Do train feat both movie and user supervise')	
	parser.add_argument('-predict',action='store_true', default=False,dest='predict',help='Do predict supervise')
	opts = parser.parse_args()
	main(opts)
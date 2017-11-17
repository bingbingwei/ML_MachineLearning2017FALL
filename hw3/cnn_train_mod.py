import csv
import sys
import numpy as np
import argparse
import keras
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, Adadelta
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils, plot_model

IM_SHAPE=48

def get_session():
	gpu_opt = tf.GPUOptions(allow_growth=True)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_opt, allow_soft_placement=True))

def read_file(train_file_path,specific):
	print(specific)
	spec = np.array(specific,dtype=int)
	train_file = open(train_file_path,'r',encoding='Big5')
	feature, label = [], []
	for index, row in enumerate(csv.reader(train_file)):
		if(index!=0):
			y = np.where(spec==(int)(row[0]))
			if y[0].size != 0:
				label.append(row[0])
				feature.append(row[1].split())
	feature = np.array(feature,dtype=int)
	label = np.array(label,dtype=int)
	print(label)
	return feature,label

def main(opts):
	ktf.set_session=(get_session())
	feature, label = [], []
	feature, label = read_file(opts.train_data_path,opts.specific.split('_'))
	feature = feature.reshape(feature.shape[0], IM_SHAPE, IM_SHAPE, 1)

	model_name = opts.model_name

	BATCH_SIZE = opts.batch_size
	EPOCH_SIZE = opts.epoch_size

	print("---------------Build Model-----------------")
	model = Sequential()

	model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
	                 input_shape=(IM_SHAPE,IM_SHAPE,1)))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))	
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))	
	
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	#model.add(Dropout(0.25))

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(output_dim=256))
	model.add(Activation('relu'))
	#model.add(BatchNormalization())
	model.add(Dense(output_dim=128))
	model.add(Activation('relu'))
	#model.add(BatchNormalization())
	model.add(Dense(output_dim=64))
	model.add(Activation('relu'))
	#model.add(BatchNormalization())
	model.add(Dense(output_dim=7))
	model.add(Activation('softmax'))
	print(model.summary())

	keras.utils.to_categorical(label, num_classes=None)

	print("-------------------------------------------")
	print("---------------Compile---------------------")
	model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	print("---------------Build End-------------------")

	callbacks = keras.callbacks.ModelCheckpoint(model_name, monitor='val_acc', verbose=0, save_best_only=True, mode='max')


	if opts.image_generate:
		valid = feature.shape[0]//10
		Valid_f = feature[:valid]
		Valid_l = label[:valid]

		datagen = ImageDataGenerator(rotation_range=10,zoom_range=0.2,fill_mode='nearest',
			width_shift_range=0.1,height_shift_range=0.1,vertical_flip=False, horizontal_flip=True)	
		datagen.fit(feature[valid:],seed=3318)
		history = model.fit_generator(datagen.flow(feature[valid:], label[valid:], batch_size=BATCH_SIZE),
                    steps_per_epoch=len(feature) / BATCH_SIZE, epochs=EPOCH_SIZE,callbacks=[callbacks],validation_data=(Valid_f,Valid_l))
		H = history.history
		np.savez('history.npz', acc=H['acc'], val_acc=H['val_acc'])
	else:
		model.fit(feature, label, batch_size=BATCH_SIZE, epochs=EPOCH_SIZE, verbose=1, validation_split=0.1, callbacks=[callbacks])
		score = model.evaluate(feature, label)
		score = '{:.6f}'.format(score[1])
		print('\nFinal train accuracy (all):', score)
		print("-------------------------------------------")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data_path', type=str,default='train.csv',dest='train_data_path',help='Training data path')
	parser.add_argument('--model_data_path', type=str,default='model.h5',dest='model_name',help='model name')
	parser.add_argument('--batch_size',type=int,default='100',dest='batch_size',help='Batch size')
	parser.add_argument('--epoch_size',type=int,default='20',dest='epoch_size',help='Epoch size')
	parser.add_argument('--specific',type=str,default='0_1_2_3_4_5',dest='specific',help='Train specific data')
	parser.add_argument('-image_generate',action='store_true', default=False,dest='image_generate',help='Add image generate')


	opts = parser.parse_args()
	main(opts)
import numpy as np
import csv
import sys
import argparse
from keras.models import load_model

def read_file(test_file_path):
	test_file = open(test_file_path,'r',encoding='Big5')
	feature = []
	for index, row in enumerate(csv.reader(test_file)):
		if(index!=0):
			feature.append(row[1].split())
	feature = np.array(feature,dtype=int)
	return feature

def main(opts):
	specific = np.array(opts.specific.split('_'),dtype=int)
	submission = open(opts.submission_data_path,"w")

	feature = []
	feature = read_file(opts.test_data_path)
	feature =feature.reshape(feature.shape[0],48,48,1)
	model = load_model(opts.model1)
	#model2 = load_model(opts.model2)
	#model3 = load_model(opts.model3)
	model4 = load_model(opts.model4)
	result = model.predict(feature, batch_size = 100, verbose = 1)
	#result2 = model2.predict(feature, batch_size = 100, verbose = 1)
	#result3 = model3.predict(feature, batch_size = 100, verbose = 1)
	writer = csv.writer(submission)
	writer.writerow(["id","label"])
	#result = result1 + result2 +result3
	result_pred = np.argmax(result,axis = 1)

	Y = []
	X = []
	map_index = []

	for index, row in enumerate(result_pred):
		y = np.where(specific==row)
		if y[0].size != 0:
			map_index.append(index)
			X.append(feature[index])
	X = np.array(X,dtype=int)
	result_spec = model4.predict(X, batch_size = 100, verbose = 1)
	result_spec_pred = np.argmax(result_spec,axis = 1)
	for index, row in enumerate(result_spec_pred):
		result_pred[map_index[index]] = row
	for index, row in enumerate(result_pred):
		writer.writerow([index,row])
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--specific',type=str,default='0_1_2_3_4_5',dest='specific',help='Train specific data')
	parser.add_argument('--model1',type=str,default='model.h5',dest='model1',help='Model 1')
	parser.add_argument('--model2',type=str,default='model.h5',dest='model2',help='Model 2')
	parser.add_argument('--model3',type=str,default='model.h5',dest='model3',help='Model 3')
	parser.add_argument('--model4',type=str,default='model.h5',dest='model4',help='Model 4')
	parser.add_argument('--test_data_path', type=str,default='test.csv',dest='test_data_path',help='Testing data path')
	parser.add_argument('--temp_data_path', type=str,default='submission_63583.csv',dest='temp_data_path',help='Temp data path')	
	parser.add_argument('--submission_data_path',type=str,default='submission_test.csv',dest='submission_data_path',help='Submission data path')
	opts = parser.parse_args()
	main(opts)

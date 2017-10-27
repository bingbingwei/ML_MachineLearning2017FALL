import csv
import numpy as np
import sys
import xgboost as xgb
import argparse


def readfile(train_data_path, train_label_path,test_data_path):
	data_file = open(train_data_path,'r',encoding='Big5')
	label_file = open(train_label_path,'r',encoding='Big5')
	test_file = open(test_data_path,'r',encoding='Big5')
	feat = []
	y = []
	test = []
	for index, row in enumerate(csv.reader(data_file)):
		if index != 0:
			feat.append(row)
	for index, row in enumerate(csv.reader(label_file)):
		if index != 0:
			y.append(row)
	for index, row in enumerate(csv.reader(test_file)):
		if index != 0:
			test.append(row)
	
	feat = np.array(feat,dtype=float)
	y = np.array(y,dtype=float)
	test = np.array(test,dtype=float)
	return feat, y, test

def writefile(ypred,submission_data_path):
	submission_file = open(submission_data_path,'w')
	writer = csv.writer(submission_file)
	writer.writerow(["id","label"])

	for index,row in enumerate(ypred):
		if(ypred[index] > 0.5):
			writer.writerow([str(index+1),"1"])
		else:
			writer.writerow([str(index+1),"0"])
def main(opts):
	data, label, test = readfile(opts.train_data_path,opts.train_label_path,opts.test_data_path)
	#XGtrain
	print(len(label))
	print(len(data))
	dtrain = xgb.DMatrix(data, label=label)
	param = {'objective':'binary:logistic'}
	evallist = [(dtrain, 'train')]
	bst = xgb.train(param,dtrain,20,evallist)
	dtest = xgb.DMatrix(test)
	ypred = bst.predict(dtest, ntree_limit=bst.best_ntree_limit)
	writefile(ypred,opts.submission_data_path)
	print(ypred)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data_path', type=str,default='X_train',dest='train_data_path',help='Training data path')
	parser.add_argument('--train_label_path', type=str,default='Y_train',dest='train_label_path',help='Training label path')
	parser.add_argument('--test_data_path',type=str,default='X_test',dest='test_data_path',help='Testing data path')
	parser.add_argument('--submission_data_path',type=str,default='submission_best.csv',dest='submission_data_path',help='Submission data path')
	parser.add_argument('-predict',action='store_true', default=False,dest='predict',help='Predict')
	opts = parser.parse_args()
	main(opts)
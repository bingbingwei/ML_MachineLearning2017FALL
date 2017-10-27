import csv
import sys
import numpy as np
import argparse

np.set_printoptions(suppress=True)
def sigmoid(z):
	res = 1/(1+np.exp(-z))
	return np.clip(res,1e-8,1-(1e-8))
def normalize(feat,feat_len):
	mean = np.mean(feat,axis=0)
	std = np.std(feat,axis=0)
	for index, row in enumerate(feat):
		feat[index] = (row-mean)/std
	feat1 = feat[:feat_len]
	test = feat[feat_len:]
	return feat1, test
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
	feat_len = len(feat)
	for index, row in enumerate(csv.reader(label_file)):
		if index != 0:
			y.append(row)
	for index,row in enumerate(csv.reader(test_file)):
		if index!= 0:
			feat.append(row)
	feat = np.array(feat,dtype=float)
	feat, test = normalize(feat,feat_len)
	y = np.array(y,dtype=float)
	return feat, y, test

def main(opts):
	feat, label, test = readfile(opts.train_data_path,opts.train_label_path,opts.test_data_path)
	submission_file = open(opts.submission_data_path,'w')
	writer = csv.writer(submission_file)
	writer.writerow(["id","label"])

	class0 = []
	class1 = []

	for index,row in enumerate(label):
		if row==0:
			class0.append(feat[index])
		else:
			class1.append(feat[index])
	class1 = np.array(class1,dtype=float)
	class0 = np.array(class0,dtype=float)
	print(class0)
	print(class1)
	N0 = len(class0)
	N1 = len(class1)
	N  = N0+N1
	mean0 = np.mean(class0, axis=0)
	mean1 = np.mean(class1,axis=0)
	cov0 = np.cov(class0, rowvar=0)
	cov1 =np.cov(class1, rowvar=0)
	cov = (N0/N)*cov0 + (N1/N)*cov1

	cov_inverse = np.linalg.pinv(cov)

	w0 = (np.dot((mean0-mean1), cov_inverse)).transpose()
	w1 = -w0
	b0 = (-0.5)*(np.dot(np.dot(mean0,cov_inverse),mean0)) + (0.5)*(np.dot(np.dot(mean1,cov_inverse),mean1)) + np.log(N0/N1)
	b1 = (0.5)*(np.dot(np.dot(mean0,cov_inverse),mean0)) + (-0.5)*(np.dot(np.dot(mean1,cov_inverse),mean1)) + np.log(N1/N0)
	
	for index,row in enumerate(test):
		z0 = np.inner(w0,row)+b0
		sig_z0 = sigmoid(z0)		
		#print("z0:"+str(sig_z0))
		z1 = np.inner(w1,row)+b1
		sig_z1 = sigmoid(z1)
		#print("z1:"+str(sig_z1))
		if(sig_z0>sig_z1):
			writer.writerow([str(index+1),"0"])
		else:
			writer.writerow([str(index+1),"1"])
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data_path', type=str,default='X_train',dest='train_data_path',help='Training data path')
	parser.add_argument('--train_label_path', type=str,default='Y_train',dest='train_label_path',help='Training label path')
	parser.add_argument('--test_data_path',type=str,default='X_test',dest='test_data_path',help='Testing data path')
	parser.add_argument('--submission_data_path',type=str,default='submission_generative.csv',dest='submission_data_path',help='Submission data path')
	opts = parser.parse_args()
	main(opts)
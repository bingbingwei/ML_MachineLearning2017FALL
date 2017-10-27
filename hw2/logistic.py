import sys
import numpy as np
import csv
import argparse

np.set_printoptions(suppress=True)
def sigmoid(z):
	res = 1/(1+np.exp(-z))
	return np.clip(res,1e-8,1-(1e-8))
def normalize(feat, feat_len):
	mean = np.mean(feat,axis=0)
	std = np.std(feat,axis=0)
	for index, row in enumerate(feat):
		feat[index] = (row-mean)/std
	feat1 = feat[:feat_len]
	test = feat[feat_len:]
	return feat1, test
def readfile(train_data_path, train_label_path, test_data_path):
	data_file = open(train_data_path,'r',encoding='Big5')
	label_file = open(train_label_path,'r',encoding='Big5')
	test_file = open(test_data_path,'r',encoding='Big5')
	feat = []
	y = []
	for index, row in enumerate(csv.reader(data_file)):
		if index != 0:
			feat.append(row)
	feat_len = len(feat)
	for index, row in enumerate(csv.reader(label_file)):
		if index != 0:
			y.extend(row)
	for index, row in enumerate(csv.reader(test_file)):
		if index != 0:
			feat.append(row)
	feat = np.array(feat,dtype=float)
	feat, test = normalize(feat,feat_len)
	print(len(feat))
	print(len(test))
	y = np.array(y,dtype=float)
	return feat,test, y
def train(feat, y, iteration, adagrad):
	w = np.zeros(feat.shape[1])
	b = 0
	l_rate = np.full(feat.shape[1],1)
	l_rate_b = 1
	sig_ada = np.zeros(feat.shape[1])
	sig_ada_b = 0
	for i in range(iteration):
		w_grad = np.zeros(feat.shape[1])
		b_grad = 0
		count = 0
		for index, row in enumerate(feat):
			z=np.dot(row,w)+b
			sig = sigmoid(z)
			w_grad -= (y[index]-sig)*row
			b_grad -= (y[index]-sig)
			if sig>0.5:
				if y[index]==1:
					count+=1
			else:
				if y[index]==0:
					count+=1
		if adagrad:
			sig_ada += w_grad*w_grad
			sig_ada_b += b_grad*b_grad
			l_rate_final = l_rate/np.sqrt(sig_ada)
			l_rate_b_final = l_rate_b/(sig_ada_b**0.5)
		else:
			l_rate_final = l_rate
			l_rate_b_final = l_rate_b
		w -= w_grad*l_rate_final
		b -= b_grad*l_rate_b_final
		print(count/feat.shape[0])
		if(i%100 == 0):
			np.save("weight_logistic",w)
			print(b)

def predict(test,submission_data_path):
	submission = open(submission_data_path,"w")
	writer = csv.writer(submission)
	writer.writerow(["id","label"])
	b = -2.15033794425
	w = np.load("weight_logistic.npy")
	w = np.array(w,dtype=float)
	print(w)
	for index, row in enumerate(test):
		dot = np.dot(row,w) + b
		sig = sigmoid(dot)
		if sig>0.5:
			writer.writerow([str(index+1),"1"])
		else:
			writer.writerow([str(index+1),"0"])


def main(opts):
	feat,test, y = readfile(opts.train_data_path, opts.train_label_path,opts.test_data_path)
	if opts.predict:
		predict(test,opts.submission_data_path)
	else:
		train(feat,y,opts.iteration,opts.adagrad)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_data_path', type=str,default='X_train',dest='train_data_path',help='Training data path')
	parser.add_argument('--train_label_path', type=str,default='Y_train',dest='train_label_path',help='Training label path')
	parser.add_argument('--test_data_path',type=str,default='X_test',dest='test_data_path',help='Testing data path')
	parser.add_argument('--iteration',type=int,default=1000,dest='iteration',help='Training iteration')
	parser.add_argument('--submission_data_path',type=str,default='submission_logistic.csv',dest='submission_data_path',help='Submission data path')
	parser.add_argument('-adagrad',action='store_true', default=False,dest='adagrad',help='Add adagrad')
	parser.add_argument('-predict',action='store_true', default=False,dest='predict',help='Predict')

	opts = parser.parse_args()
	main(opts)
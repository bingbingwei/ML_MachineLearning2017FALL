import csv
import sys
import numpy as np

allFeature = []
testFeature = []

def Readfile():
	global allFeature
	global testFeature
	TrainFile = open('train.csv','r',encoding='Big5')
	TestFile = open('test_X.csv','r',encoding='Big5')
	for index,row in enumerate(csv.reader(TrainFile)):
		if (index%18) == 10:
			row = row[3:]
			allFeature.extend(row) 
	for index, row in enumerate(csv.reader(TestFile)):
		if (index%18) == 9:
			row = row[2:]
			testFeature.append(row)
	allFeature =  np.array(allFeature, dtype = np.float)
	testFeature =  np.array(testFeature, dtype = np.float)

def Train():
	global allFeature
	global testFeature
	feature = []
	y = []
	gradient = np.zeros(7)
	loss = 0
	diff = 0
	learningRate = np.full(7,0.00000002)
	learningRateBias = 0.00001
	w = np.zeros(7)
	b = 0

	gradientBias = 0
	for i in range(len(allFeature)-7):
		if (i%480) <473:
			feature.append(allFeature[i:i+7])
			y.extend([allFeature[i+7]])
	trainSize = len(feature)
	print(trainSize)
	for row in testFeature:
		feature.append(row[0:7])
		feature.append(row[0:7])
		feature.append(row[0:7])
		feature.append(row[1:8])
		feature.append(row[1:8])
		feature.append(row[1:8])
		y.extend([row[7]])
		y.extend([row[7]])
		y.extend([row[7]])
		y.extend([row[8]])
		y.extend([row[8]])
		y.extend([row[8]])
	print(len(feature))

	for i in range(10000):
		count = 0
		for index,row in enumerate(feature):
			dot = np.dot(row,w) + b
			#print("dot:"+str(dot))
			#print("y  :" +str(allFeature[index+7]))
			#if index<trainSize:
			gradient -= (2*(y[index]-dot)*row)
			#print("row:"+str(row))
			#print("gra:"+str(gradient))
			gradientBias -= (2*(y[index]-dot))
			loss += (y[index]-dot)**2

		w -= learningRate*gradient
		b -= learningRateBias*gradientBias
		gradient = np.zeros(7)
		gradientBias = 0
		print(w)
		print("b="+str(b))
		print("loss:"+str((loss/len(feature))**0.5))
		loss = 0


def main():
	Readfile()
	Train()



if __name__ == "__main__":
	main()
import sys
import numpy as np
import csv

def Readfile():
	PMdata = []
	TrainFile = open(sys.argv[2],'r',encoding='Big5')
	i = int(sys.argv[1])
	count = 11-i
	submission = open(sys.argv[3],"w")

	for index,row in enumerate(csv.reader(TrainFile)):
		if(index%18 == 9):
			row = row[count:]
			PMdata.append(row)
			
	w = [ 0.13882413, -0.23826578 ,-0.05202318,  0.55701603, -0.61302175, -0.03094573, 1.16405781]  #7.059
	b = 1.76465280244

	w = np.array(w,dtype= float)
	writer = csv.writer(submission)
	writer.writerow(["id","value"])
	for index, row in enumerate(PMdata):
		row = np.array(row,dtype = float)
		ans = np.dot(w,row)+b
		writer.writerow(["id_"+str(index),str(ans)])



def main():
	Readfile()

if __name__ == "__main__":
	main()
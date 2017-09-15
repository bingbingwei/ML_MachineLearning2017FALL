import sys

def main():
	file1 = open(sys.argv[1],"r")
	file2 = open("Q1.txt","w")
	string = ""
	dict = {}
	for line in file1:
		words = line.split()
		for word in words:
			if dict.get(word,"NotFound") == "NotFound":
				dict[word] = 1;
			else:
				dict[word] = int(dict[word])+1;
	count = 0
	for key, value in dict.items():
		string = string + (str(key)+" "+str(count)+" "+str(value)+"\n")
		count =count + 1
	string = string[:-1]
	file2.write(string)

if __name__ == "__main__":
	main()
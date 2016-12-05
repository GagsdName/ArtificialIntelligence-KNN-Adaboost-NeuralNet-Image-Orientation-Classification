import sys, math

train_dict={}

#Nearest Neighbor Classifier
def nearestNeighbor(testFileName):
	f = open(testFileName)
	for line in f:
		lineTokens = line.split()
		vector = lineTokens
	
	 	min_val = 999999999999
         	nearest = ""
         	euc_sum = 0

		for key in train_dict:
			size_train = len(train_dict[key]["vector"])
			size_test = len(lineTokens[2::])
			size = 0
			if size_test < size_train:
				size = size_test
			else: size = size_train
		
			euc_sum = 0
			for k in range(size):
				train_veck = int(train_dict[key]["vector"][k])
				test_veck = int(lineTokens[2::][k])
				
				diff = math.fabs(test_veck - train_veck)
				euc_sum = euc_sum + diff ** 2
			euclidean = math.sqrt(euc_sum)
			#print euclidean
			if euclidean < min_val:
				min_val = euclidean
				nearest = str(key)		 
	
		print "Nearest Neighbor for  - ", lineTokens[0], " with orientation - ", lineTokens[1], " is - ",\
			 nearest, " with orientation - ", train_dict[nearest]["orientation"]
	
#read training file
def readTrainFile(filename):
	f = open(filename, 'r')
        for line in f:
		lineTokens = line.split()
		photo_id = str(lineTokens[0])
		orientation = int(lineTokens[1])
		vector = lineTokens[2::]
		train_dict.update({photo_id:{"orientation":orientation, "vector":vector}})
		#print vector
		


input = sys.argv[1:4] #input arguments
if len(input) == 3: #check to see if correct number of arguments are there
        train_file = input[0]
        test_file = input[1]
        mode = input[2]
	readTrainFile(str(train_file))
	if mode == "nearest":
		nearestNeighbor(str(test_file))
else:
        print "enter all input parameters!"



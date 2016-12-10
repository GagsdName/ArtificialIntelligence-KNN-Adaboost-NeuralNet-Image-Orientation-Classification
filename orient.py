import sys, math, operator

train_dict={}

#Nearest Neighbor Classifier
def nearestNeighbor(testFileName):
	f = open(testFileName,'r')
	f1 = open('nearest_output.txt', 'w')
	total = 0
	correct = 0
	w = 4 #length and width of the confusion matrix - given assumption in problem statement - total number of topics is 20
        conf_mtr = [[0 for x in range(w)] for y in range(w)] #intializing confusion matrix
	Labels = {"0":0,"90":1,"180":2,"270":3}	
	for line in f:
		total+=1
		lineTokens = line.split()
		vector = lineTokens
		kvalue = 2 #k value for k neighbors
		knearest={}
		for i in range(kvalue):	
	 		min_val = 999999999999
         		euc_sum = 0
			nearest = ""
			for key in train_dict:
				if key not in knearest:
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
				#euclidean = math.sqrt(euc_sum)
				#print euclidean
					if euc_sum < min_val:
						min_val = euc_sum
						nearest = str(key)		 
	
			#print "Nearest Neighbor for  - ", lineTokens[0], " with orientation - ", lineTokens[1], " is - ",\
			#	 nearest, " with orientation - ", train_dict[nearest]["orientation"]
			#f1.write(str(lineTokens[0])+" "+str(train_dict[nearest]["orientation"])+"\n")
			knearest.update({nearest:train_dict[nearest]["orientation"]})
		#knearest.sort(key=operator.itemgetter(1))
		sortedKneighbors = sorted(knearest.iteritems(), key=operator.itemgetter(1), reverse=True)
		if int(lineTokens[1]) != sortedKneighbors[0][1]:
			conf_mtr[Labels[str(lineTokens[1])]][Labels[str(sortedKneighbors[0][1])]] +=1
	
		print kvalue, "Nearest Neighbor for  - ", lineTokens[0], " with orientation - ", lineTokens[1], " is - ",\
			 sortedKneighbors[0][0], " with orientation - ", sortedKneighbors[0][1]
		
		f1.write(str(lineTokens[0])+" "+str(sortedKneighbors[0][1])+"\n")
        #printing confusion matrix
        incorrect = 0
	for x in range(4):
                temp = ""
                for y in range(4):
                        if int(conf_mtr[x][y])!=0:
				incorrect+=conf_mtr[x][y]
			
			temp = temp + str(conf_mtr[x][y])+"\t"
                print temp+"\n"
		
	print "\nAccuracy = ", float(total - incorrect)/total, "%"
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



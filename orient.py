from __future__ import division
import sys, math
import numpy as np
from artificial_nn import NeuralNet
# ************************************** Knn ************************************************

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


# ************************************** Adaboost ************************************************
stump_dict={}

class classifier(object):
	comparator_one = None
	comparator_two = None
	alpha = None
	error = None
	orientation = None

	


def train_adaboost(stump):
	create_stumps(stump)
	return 

#create stumps for adaboost
def create_stumps(stump):
	orientation = [0, 90, 180, 270]
	for i in orientation:
		create_stumps_for_orientation(i, stump)
	return 

#create n stumps for the given orientation 	
def create_stumps_for_orientation(orientation, stump):
	number_of_pics = len(train_dict.keys())
	weights = {}
	#initial weights
	for i in train_dict:
		weights.update({i:1.0/number_of_pics})
	
	#initial right and wrong lists
	for i in range(int(stump)):
		error = 0.0
		right = []
		wrong = []

		classifier_object = classifier()
		
		#generate random comparator_one
		classifier_object.comparator_one = np.random.choice(192)

		#generate random comparator_two
		while True:
			classifier_object.comparator_two = np.random.choice(192)
			if classifier_object.comparator_one != classifier_object.comparator_two:
				break

		classifier_object.orientation = orientation
		for pic in train_dict:
			if not classify_image(train_dict[pic][orientation], classifier_object):
				error += weights[pic]
				wrong.append(pic)
			else:
				right.append(pic)
		
		classifier_object.error = error

		#calculate wrong weights
		for keys in weights:
			if keys in right:
				weights.update({keys:0.5/len(right)})
			else:
				weights.update({keys:0.5/len(wrong)})

		classifier_object.alpha  = 0.5*np.log((1.0-error)/error)

		if orientation in stump_dict:
			stump_dict[orientation].append(classifier_object)
		else:
			stump_dict[orientation] = [classifier_object]

		print "Orientation: ",
		print orientation
		print "Stump: ",
		print i+1
		print classifier_object.comparator_one
		print classifier_object.comparator_two
		print classifier_object.orientation
		print error
		print classifier_object.alpha
		print "************"
	return


def run_adaboost_test():
	correct = 0
	total = 0
	for i in test_dict:
		orientation = test_dict[i].keys()[0]
		vector = test_dict[i].values()[0]
		predicted_orientation = get_orientation(vector)
		print "Topic: ",
		print orientation,
		print "		Topic Predicted: ",
		print predicted_orientation
		if predicted_orientation == orientation:
			correct += 1
		total += 1

	print "Accuracy: ",
	print correct*100.0/total

	print correct
	print total

def get_orientation(vector):
	orientation_values = {}
	for orientation in stump_dict:
		value = 0.0
		classifiers = stump_dict[orientation]
		for stump in classifiers:
			if classify_image(vector, stump):
				value += stump.alpha
			else:
				value -= stump.alpha
		orientation_values.update({orientation : value})
	return max(orientation_values, key=orientation_values.get)

#classify image based on stump
def classify_image(image, c):
	if image[c.comparator_one] > image[c.comparator_two]:
		return True
	return False

# ************************************** Neural nets ************************************************


# ************************************** Common *********************************************
train_dict={}
test_dict={}

#read training file
def readTestFile(filename):
	print('Reading test file...')
	f = open(filename, 'r')
	for line in f:
		lineTokens = line.split()
		photo_id = str(lineTokens[0])
		orientation = int(lineTokens[1])
		vector = lineTokens[2::]
		if photo_id not in train_dict:
			test_dict[photo_id] = {}
		test_dict[photo_id][orientation] = vector
	f.close()
	print('Reading test file complete!')

#read training file
def readTrainFile(filename):
	print('Reading training file...')
	f = open(filename, 'r')
	for line in f:
		lineTokens = line.split()
		photo_id = str(lineTokens[0])
		orientation = int(lineTokens[1])
		vector = lineTokens[2::]
		if photo_id not in train_dict:
			train_dict[photo_id] = {}
		train_dict[photo_id][orientation] = vector
	f.close()
	print('Reading training file complete!')

inputArg = sys.argv[1:5] #input arguments
if len(inputArg) < 4: #check to see if correct number of arguments are there
	print "enter all input parameters!"
	exit()
train_file = inputArg[0]
test_file = inputArg[1]
mode = inputArg[2]
stump = inputArg[3]
readTrainFile(str(train_file))
readTestFile(str(test_file))

if mode == "adaboost":
	train_adaboost(stump)
	run_adaboost_test()

if mode == 'nnet':
	# stump = number of neurons in hidden layer
	# Output classes = 4 ==> Number of neurons in outputLayer = 4
	# Number of neurons in inputLayer = length of feature vector = 192
	# Set-up the Neural Net
	learningRate = float(sys.argv[-1]) if len(sys.argv)>4 else 1.5
	nnet = NeuralNet()
	nnet.initializeNN(192, int(stump), 4)
	print("{}:{}".format('Learning Rate', learningRate))
	nnet.train_Network(train_dict, learningRate, 20, 4)
	# nnet.printNN()
	# Run the trained neural network classifier on test data
	totalCount = 0
	correctCount = 0
	for photoId in test_dict:
		for orientation in test_dict[photoId]:
			totalCount += 1
			# Convert the string array to int array
			# Dividing each input by 500 to scale it down and get a small activation value
			inputVector = [(int(x)/500.0) for x in test_dict[photoId][orientation]]
			prediction = nnet.predict(inputVector)
			if prediction == orientation:
				correctCount += 1
	print("{}:{}".format('Neural Network Accuracy:', correctCount/totalCount))
	
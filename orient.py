from __future__ import division
import sys, math, operator
import numpy as np
from artificial_nn import NeuralNet
# ************************************** Knn ************************************************

#Nearest Neighbor Classifier
def nearestNeighbor():
	w = 4 #length and width of the confusion matrix - given assumption in problem statement - total number of topics is 20
        conf_mtr = [[0 for x in range(w)] for y in range(w)] #intializing confusion matrix
        Labels = {"0":0,"90":1,"180":2,"270":3}
        f1 = open('nearest_output.txt', 'w')
	correct = 0
	total = 0
	for pic in test_dict.values():
		orientation = pic.keys()[0]
		vector = pic.values()[0]
		v1 = np.array(vector)
		min_distance = sys.float_info.max
		for train_pics in train_dict.values():
			for i in range(len(train_pics)):
				temp_orientation = train_pics.keys()[i]
				temp_vector = train_pics.values()[i]
				v2 = np.array(temp_vector)
				distance = np.linalg.norm(v2-v1)
				if distance < min_distance:
					orientation_predicted = temp_orientation
					min_distance = distance
		print "Image ID: "+ test_dict.keys()[test_dict.values().index(pic)] + "    Orientation: " + str(orientation) +\
                 "              Predicted: " + str(orientation_predicted)
                f1.write(str(test_dict.keys()[test_dict.values().index(pic)])+" "+str(orientation_predicted)+"\n")
		if orientation_predicted == orientation:
			correct += 1
		else: conf_mtr[Labels[str(orientation)]][Labels[str(orientation_predicted)]] +=1
		total += 1
	
	#printing confusion matrix
        for x in range(4):
        	temp = ""
                for y in range(4):
                        temp = temp + str(conf_mtr[x][y])+"\t"
                print temp+"\n"


	print "Accuracy: ",
	print str(correct*100.0/total)+"%"
	return
# ************************************** Adaboost ************************************************
stump_dict={}

class classifier(object):
	comparator_one = None
	comparator_two = None
	alpha = None
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
	for pic in train_dict:
		weights.update({pic:1.0/number_of_pics})
	
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

		#calculate wrong weights
		for pic in right:
			weights.update({pic:0.5/len(right)})
		for pic in wrong:
			weights.update({pic:0.5/len(wrong)})

		#calculate alpha
		classifier_object.alpha  = 0.5*np.log((1.0-error)/error)

		if orientation in stump_dict:
			stump_dict[orientation].append(classifier_object)
		else:
			stump_dict[orientation] = [classifier_object]
	return

def run_adaboost_test():
	correct = 0
	total = 0
	correct_labels = []
	predicted_labels = []
	f1 = open('adaboost_output.txt', 'w')
	for i in test_dict:
		orientation = test_dict[i].keys()[0]
		vector = test_dict[i].values()[0]
		predicted_orientation = get_orientation(vector)
		print "Orientation: ",
		print orientation,
		print "		Orientation Predicted: ",
		print predicted_orientation
	
		if predicted_orientation == orientation:
			correct += 1
		total += 1
		correct_labels.append(orientation)
		predicted_labels.append(predicted_orientation)
		f1.write(str(i)+" "+str(predicted_orientation)+"\n")

	printConfusionMatrix(create_conf_matrix(correct_labels, predicted_labels, 4))
	print "Accuracy: ",
	print correct*100.0/total

	return

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
		test_dict[photo_id][orientation] = map(int, vector)
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
		train_dict[photo_id][orientation] = map(int, vector)
	f.close()
	print('Reading training file complete!')

# Creates the Confusion Matrix	
def create_conf_matrix(expected, predicted, n_classes):
	toIndexTransformer = {0:0, 90:1, 180:2, 270:3}
	m = [[0] * n_classes for _ in range(n_classes)]
	for pred, exp in zip(predicted, expected):
		m[toIndexTransformer[pred]][toIndexTransformer[exp]] += 1
	return m

# Calculates the accuracy from the confusion matrix
def calc_accuracy(confMatrix):
	t = sum(sum(l) for l in confMatrix)
	return sum(confMatrix[i][i] for i in range(len(confMatrix))) / t

# Prints the Confusion Matrix
def printConfusionMatrix(confMatrix):
	print('Confusion Matrix')
	print(' | '.join([str(i) for i in [0, 90, 180, 270]]))
	print('--------------------')
	for row in confMatrix:
		print(row)

inputArg = sys.argv[1:5] #input arguments
if len(inputArg) < 3: #check to see if correct number of arguments are there
	print "enter all input parameters!"
	exit()
train_file = inputArg[0]
test_file = inputArg[1]
mode = inputArg[2]
readTrainFile(str(train_file))
readTestFile(str(test_file))

if mode == "nearest":
	 nearestNeighbor()
	
if mode == "adaboost":
	stump = inputArg[3]
	train_adaboost(stump)
	run_adaboost_test()

if mode == 'nnet' or mode == 'best':
	# stump = number of neurons in hidden layer
	# Output classes = 4 ==> Number of neurons in outputLayer = 4
	# Number of neurons in inputLayer = length of feature vector = 192
	# Set-up the Neural Net
	hiddenCount = int(inputArg[3]) if mode=='nnet' else 4
	learningRate = 3.6
	epochs = 25
	nnet = NeuralNet()
	nnet.initializeNN(192, hiddenCount, 4)
	# print("{}:{}".format('Learning Rate', learningRate))
	nnet.train_Network(train_dict, learningRate, epochs, 4)
	# nnet.printNN()
	# Run the trained neural network classifier on test data
	correctLabels = []
	predictedLabels = []
	result = open('nnet_output.txt', 'w')
	for photoId in test_dict:
		for orientation in test_dict[photoId]:
			# Convert the string array to int array
			# Dividing each input by 500 to scale it down and get a small activation value
			inputVector = [(int(x)/500.0) for x in test_dict[photoId][orientation]]
			prediction = nnet.predict(inputVector)
			result.write(photoId + ' ' + str(prediction) + '\n')
			predictedLabels.append(prediction)
			correctLabels.append(orientation)
	result.close()
	confMatrix = create_conf_matrix(correctLabels, predictedLabels, 4)
	printConfusionMatrix(confMatrix)
	print("{}:{}".format('Classification Accuracy', calc_accuracy(confMatrix)))

'''
@author: Manan
'''
from __future__ import division
from random import random, seed
from math import exp
# This class represents the Artificial Neural Network
class NeuralNet:
    def __init__(self):
        # The network is maintained as a list of layers (Input, Hidden and Output). Each layer will consist of neurons,
        # represented by a dictionary. So effectively, each layer will be a list of dictionaries.
        # We don't have an actual input layer below, since it is nothing but the feature vector itself.
        self.network = []
        # This dict will be used during training to transform orientation value to an index value for
        # expected output vector
        self.toIndexTransformer = {0:0, 90:1, 180:2, 270:3}
        self.toOrientationTransformer = {0:0, 1:90, 2:180, 3:270}
    
    # setNetwork allows us to set-up an existing network and use it for testing purpose
    def setNetwork(self, network):
        self.network = network
    # initializeNN accepts three parameters as input.
    # 1) input_n = number of input features = length of feature vector
    # 2) hidden_n = number of neurons in hidden layer
    # 3) output_n = number of outputs
    def initializeNN(self, input_n, hidden_n, output_n):
        seed(1)
        hiddenLayer = []
        outputLayer = []
        # Initialize the neurons in Hidden and Output layer with random weights
        # In the hidden layer, the length of weight list within each neuron will be input_n+1
        # The extra weight at the end is assumed to be for bias.
        # Similarly, in the Output Layer, the length of weight list within each neuron will be hidden_n+1
        # The extra weight at the end is assumed to be for bias.
        for _ in range(hidden_n):
            hiddenLayer.append({'weights':[random() for _ in range(input_n+1)]})
        for _ in range(output_n):
            outputLayer.append({'weights':[random() for _ in range(hidden_n+1)]})
        self.network.append(hiddenLayer)
        self.network.append(outputLayer)
        print('Network initialized!')
        print(self.network)
        
    # activateNeuron is the activation function which computes activation value
    # Input: 1) neuron to be activated 2) list of input values
    def activateNeuron(self, neuron, inputParam):
        weights = neuron['weights']
        bias = weights[-1] # The last weight value is assumed to be for bias
        return (bias + (sum([weights[i]*inputParam[i] for i in range(0,len(inputParam))])))
    
    # Neuron transfer activation using Sigmoid function
    def transferActivation(self, activation):
        return 1.0 / (1.0 + exp(-activation))
    
    # forwardPropogate method carry-forwards the output of one layer to the next layer.
    # Output of any layer serves as input to the next layer
    # Initially, this function will be called to propagate the inputLayer values to hidden layer
    def forwardPropogate(self, output):
        # print("{}-{}".format('Input Values to be propagated', output))
        inputValues = output
        for i in range(len(self.network)):
            newInputValues = []
            for neuron in self.network[i]:
                activation = self.activateNeuron(neuron, inputValues)
                neuron['outputVal'] = self.transferActivation(activation)
                newInputValues.append(neuron['outputVal'])
            inputValues = newInputValues
        # The returned inputValues will be the output values of neurons from the outputLayer
        # print("{}-{}".format('Output Values', inputValues))
        return inputValues
    
    # calculateDerivative method calculates the derivative of the output value of a neuron
    def calculateDerivative(self, neuron):
        outputVal = neuron['outputVal']
        return (outputVal * (1.0-outputVal))
    
    # backPropagateError method back-propagates the error starting from the outputLayer in reverse direction
    def backPropagateError(self, expected):
        # Calculate the error for the outputLayer first
        outputLayer = self.network[-1]
        for i in range(len(outputLayer)):
            neuron = outputLayer[i]
            actualOutput = neuron['outputVal']
            neuron['error'] = (expected[i]-actualOutput)*self.calculateDerivative(neuron)
        # Now start from the penultimate layer in reverse direction
        for i in reversed(range(len(self.network)-1)):
            currentLayer = self.network[i]
            # Calculate the error value for each neuron in currentLayer
            for j in range(len(currentLayer)):
                currentNeuron = currentLayer[j]
                error = 0.0
                for neuron in self.network[i+1]:
                    error += neuron['weights'][j] * neuron['error']
                currentNeuron['error'] = error * self.calculateDerivative(currentNeuron)
    
    # updateNeuronWeights method updates the weight of each neuron in the network wrt. the input values
    # The input for hiddenLayer is the set of input from inputLayer. The output from hiddenLayer will
    # serve as input for the outputLayer.
    def updateNeuronWeights(self, inputValues, learningRate):
        for i in range(len(self.network)):
            inputSet = inputValues
            # if current layer is not 1st layer, then update the inputSet
            if i>0:
                inputSet = [neuron['outputVal'] for neuron in self.network[i-1]]
            for neuron in self.network[i]:
                # print("{}-{}".format('Before Update', neuron))
                # Update all weights except the bais-weight in current neuron
                for j in range(len(neuron['weights'])-1):
                    neuron['weights'][j] += learningRate * neuron['error'] * inputSet[j]
                # Update the bias-weight
                neuron['weights'][-1] += learningRate * neuron['error']
                # print("{}-{}".format('After Update', neuron))
       
    # train_Network method is called from the main script - orient.py to train the Neural Net from training data
    # 1) trainingData = training data read from training file and passed as a dict
    # 2) learningRate = Rate of learning for the network
    # 3) epoch = Number of training iterations ever training data
    # 4) outputClassCount = Number of possible output values = 4
    def train_Network(self, trainingData, learningRate, epochs, outputClassCount):
        for i in range(epochs):
            print("{}-{}".format('Current Epoch', i))
            # Calculate the Squared Error for each epoch
            error = 0.0
            for photoId in trainingData:
                for orientation in trainingData[photoId]:
                    # Convert the string array to int array
                    # Dividing each input by 500 to scale it down and get a small activation value
                    inputVector = [(int(x)/500.0) for x in trainingData[photoId][orientation]]
                    networkOutput = self.forwardPropogate(inputVector)
                    # print("{}-{}".format('Actual Output', networkOutput))
                    # Set the expected output vector
                    expectedOutput = [0] * outputClassCount
                    expectedOutput[self.toIndexTransformer[orientation]] = 1
                    # print("{}-{}".format('Expected Output', expectedOutput))
                    for j in range(len(expectedOutput)):
                        error += (expectedOutput[j]-networkOutput[j])**2
                    # Back-propagate the error
                    self.backPropagateError(expectedOutput)
                    # Update the weights of each neuron in network
                    self.updateNeuronWeights(inputVector, learningRate)
            print("{}-{}".format('Squared error for current epoch', error))
            print(self.network)
        print('Training Complete! Below is the final network configuration.')
        
    def predict(self, featureVector):
        outputValues = self.forwardPropogate(featureVector)
        return self.toOrientationTransformer(outputValues.index(max(outputValues)))
    
    def printNN(self):
        for layer in self.network:
            print(layer)                
                    
                    
"""nnTest = NeuralNet()
# test forward propagation
network = [[{'outputVal': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
        [{'outputVal': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'outputVal': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
nnTest.setNetwork(network)
expected = [0, 1]
nnTest.backPropagateError(expected)
for layer in nnTest.network:
    print(layer)"""
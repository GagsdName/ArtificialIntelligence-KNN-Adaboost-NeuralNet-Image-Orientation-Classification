'''
@author: Manan
'''
from __future__ import division
from random import random
from math import exp
# This class represents the Artificial Neural Network
class NeuralNet:
    def __init__(self):
        # The network is maintained as a list of layers (Input, Hidden and Output). Each layer will consist of neurons,
        # represented by a dictionary. So effectively, each layer will be a list of dictionaries.
        # We don't have an actual input layer below, since it is nothing but the feature vector itself.
        self.network = []
    
    # setNetwork allows us to set-up an existing network and use it for testing purpose
    def setNetwork(self, network):
        self.network = network
    # initializeNN accepts three parameters as input.
    # 1) input_n = number of input features = length of feature vector
    # 2) hidden_n = number of neurons in hidden layer
    # 3) output_n = number of outputs
    def initializeNN(self, input_n, hidden_n, output_n):
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
        inputValues = output
        for layer in self.network:
            newInputValues = []
            for neuron in layer:
                activation = self.activateNeuron(neuron, inputValues)
                neuron['outputVal'] = self.transferActivation(activation)
                newInputValues.append(neuron['outputVal'])
            inputValues = newInputValues
        # The returned inputValues will be the output values of neurons from the outputLayer
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

"""nnTest = NeuralNet()
# test forward propagation
network = [[{'outputVal': 0.7105668883115941, 'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}],
        [{'outputVal': 0.6213859615555266, 'weights': [0.2550690257394217, 0.49543508709194095]}, {'outputVal': 0.6573693455986976, 'weights': [0.4494910647887381, 0.651592972722763]}]]
nnTest.setNetwork(network)
expected = [0, 1]
nnTest.backPropagateError(expected)
for layer in nnTest.network:
    print(layer)"""
'''
@author: Manan
'''
from random import random
# This class represents the Artificial Neural Network
class NeuralNet:
    def __init__(self):
        # The network is maintained as a list of layers (Input, Hidden and Output). Each layer will consist of neurons,
        # represented by a dictionary. So effectively, each layer will be a list of dictionaries.
        # We don't have an actual input layer below, since it is nothing but the feature vector itself.
        self.network = []
    
    # initializeNN accepts three parameters as input.
    # 1) input_n = number of input features = length of feature vector
    # 2) hidden_n = number of neurons in hidden layer
    # 3) output_n = number of outputs
    def initializeNN(self, input_n, hidden_n, output_n):
        hiddenLayer = []
        outputLayer = []
        # Initialize the neurons in Hidden and Output layer with random weights
        for _ in range(hidden_n):
            hiddenLayer.append({'weights':[random() for _ in range(input_n+1)]})
        for _ in range(output_n):
            outputLayer.append({'weights':[random() for _ in range(hidden_n+1)]})
        self.network.append(hiddenLayer)
        self.network.append(outputLayer)
        

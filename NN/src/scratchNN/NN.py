"""
NN.py
~~~~~~~~~

module to build neural net
"""

import numpy as np

class Layer():

    ''' class to represent a dense layer in a neural network 

        includes weights and biases for each node in the layer
    '''

    types = ['input', 'hidden', 'output']
    activation_functions = ['relu', 'softmax']

    def __init__(self, type, num_nodes, activation_function, input_data=None, prev_layer=None):
        ''' initialize the layer with random weights and biases

        Parameters:
            type (str): the type of layer (input, hidden, output)
            num_nodes (int): the number of nodes in the layer
            num_inputs (int): the number of inputs to each node 
                              corresponds to number of rows in previous layer
        '''
        if type not in self.types:
            raise ValueError(f"Layer type must be one of {self.types}")
        else:
            self.type = type

        if activation_function not in self.activation_functions:
            raise ValueError(f"Activation function must be one of {self.activation_functions}")

        if prev_layer:
            num_inputs = prev_layer.num_nodes

        if type == 'input':
            self.weights = None
            self.biases = None
            self.activation_function = None
            self.activations = input_data
        else:
            self.weights = np.random.randn(num_nodes, num_inputs)
            self.biases = np.random.randn(num_nodes, 1)
            self.activation_function = activation_function
            self.activations = np.zeros((num_nodes, prev_layer.activations.shape[1]))

    def Relu(self, Z):
        return np.maximum(0,Z)
    
    def Softmax(self, Z):
        return np.exp(Z) / np.sum(np.exp(Z))

    

class NeuralNet():
    ''' class to represent network architecture and perform forward and backward propagation
    '''

    def __init__(self, layers):
        ''' initialize the network with the layers

            Parameters:
                layers (list): list of Layer objects in order from input to output
        '''
        self.layers = layers
        self.activations = list(len(layers))

    def Forward(self, layer1, layer2):
        ''' Forward pass through the layer to determine the activation
            of the current layer

            Parameters: prev_layer_data (ndarray): 
                data or activations from the previous layer
        '''
        Z = np.dot(layer2.weights, layer1.activations) + layer2.biases
        if layer2.activation_function == 'relu':
            layer2.activations = layer2.Relu(Z)
        elif layer2.activation_function == 'softmax':
            layer2.activations = layer2.Softmax(Z)

    def Loss(self, label_data, output_layer):
        ''' Calculate the loss of the network

            Parameters:
                label_data (ndarray): the true labels of the data
                output_data (ndarray): the output of the network

            Returns:

        '''
        return output_layer.activations - label_data

    def Backward(self, label_data, prev_layer_data, prev_layer_weights):
        ''' Backward pass through the layer to determine the gradient
            of the loss with respect to the weights and biases

            Parameters: 
                prev_layer_data (ndarray): data or activations from the previous layer
                next_layer_data (ndarray): data or activations from the next layer
                next_layer_weights (ndarray): weights from the next layer
        '''
        if self.type == 'input':
            return None
        if self.type == 'output':
            dZ = label_data - self.
            dZ = next_layer_data
            dW = np.dot(dZ, prev_layer_data.T)
            db = dZ
            return dW, db
        else:
            dZ = np.dot(next_layer_weights.T, next_layer_data)
            dW = np.dot(dZ, prev_layer_data.T)
            db = dZ
            return dW, db


    
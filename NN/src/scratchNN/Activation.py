from abc import ABC, abstractmethod
import numpy as np

# treat activation functions as separate layers
    # allows you to easily stack different operations when building the network


class Activation(ABC):
    # don't need an init because most act functions don't need to initialize any attributes upon instance creation
    # activation functions operate statelessly on their parameters
        # they don't retain or update internal state across calls
        # their behaviour is entirely determined by their input at any given call
        # have no parameters to be updated during training

    @abstractmethod
    def forward(self, input):
        """
            Apply activation function on the input
            :param input: output from teh previous layer
            :return: the activated function
        """
        pass

    @abstractmethod
    def backward_step(self, grad_output):
        """
            Compute the gradient of the activation function w.r.t. the input
            :param input: input to the activation function
            :param grad_output: the gradient of the loss w.r.t the output of the activation function
            :return: the gradient of the loss w.r.t the input of the activation function
        """
        pass


class ReLU(Activation):
    # ReLU introduces non-linearity and leads to sparsity in the NN's activations
        # i.e. some neurons will not activate
    
    def forward(self, input):
        """
            Apply ReLU act function: max(0, input)
            :param input: input array from previous layer
            :return: output array after applying ReLU    
        """
        self.input = input
        return np.maximum(0, input)
    
    def backward_step(self, grad_output):
        """
            Compute the gradient of the ReLU function w.r.t the input
            :param input: input to the ReLU function
            :param grad_output: the gradient of the loss w.r.t the output of the ReLU function
            :return: the gradient of the loss w.r.t the input of the ReLU function
        """
        # ReLU derivative is 1 if input > 0, 0 otherwise
        print("  ReLU back step")
        print("    grad_output shape: ", grad_output.shape)
        print("    input shape: ", self.input.shape) 
        return (self.input > 0) * grad_output
    
# !!! works only for classification problems with one-hot encoded labels !!!
# This softmax implementation is only suitable for the output layer of a classification network
class SoftMax(Activation):
    # Softmax converts output activations (logits) to a probability distribution
    # each row of input array represents a set of logits for a single data point
    # (for output layer) input array has size (batch_size, num_classes)

    def forward(self, input):
        """
            Apply Softmax act function: exp(Z) / sum(exp(Z))
            :param input: input array from previous layer
            :return: output array (probability dist) after applying softmax    
        """
        # axis=1 corresponds to summing across all logits for each individual data instance
        # keepdims=Trueensures that the output of the sum retains the same dimensions as the input
            # ensures division is properly broadcast over each row
        self.input = input
        return np.exp(input) / np.sum(np.exp(input), axis=1, keepdims=True)
    
    def backward_step(self, grad_output):
        """
            Compute the gradient of the Softmax function w.r.t the input
            :param input: same as input for forward pass
            :param grad_output: corresponds to one-hot encoded labels 
            :return: the gradient of the loss w.r.t the input of the SoftMax function
        """
        print("  SoftMax back step")
        print("    grad_output shape: ", grad_output.shape)
        print("    input shape: ", self.input.shape)
        return self.input - grad_output
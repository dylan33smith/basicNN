# ABC module is used to define abstract classes
    # @abstractmethod decorator is used to define abstract methods
    # Abstract methods are methods that are declared but contains no implementation
    # any subclass of 'Layer' must implement these methods
from abc import ABC, abstractmethod
import numpy as np

# treat activation functions as separate layers
    # allows you to easily stack different operations when building the network

# updating weights is handled by an optimizer which adjusts the weights based on gradients calculated during back prop


class Layer(ABC):
    def __init__(self):
        # constructor for abstract class initializes common attributes

        # trainable attribute is used to determine if the layer is trainable
        self.trainable = True
    
    @abstractmethod
    def forward(self, input):
        """
             Compute the ouput of the layer given input
             :param input: Indput data or output from the previous layer
             :return: Layer output
        """
        pass
    
    @abstractmethod
    def backward_step(self, grad_output):
        """
            back propagate the gradient through the layer
            :param grad_output: The gradient of the loss with respect to the output of the layer
            :return: The gradient of the loss with respect to the input of the layer
        """
        pass

    @abstractmethod
    def get_parameters_and_gradients(self):
        """
            Get the parameters of the layer and their gradients
            :return: A list of tuples where each tuple contains a reference to a parameter (weight or bias) and its corresponding gradient
                for layers without trainable parameters, this method should return an empty list
        """
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        """
            Constructor for the Dense layer
            :param input_size: The number of input neurons
            :param output_size: The number of output neurons
            
        """
        # call the constructor of the parent class
        # calling super() ensuress initialization of base class ('Layer') is called
        # important for proper inheritance
        # ensures base class is correctly initialized before subclass adds specific initializations
        super().__init__()
        # initialize the weights and biases
        # each input node is connected to each output node 
            # number of weights = input_size * output_size
        # weights are input weights
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.random.randn(output_size)

    def forward(self, input):
        """
            Compute the output of the layer given input
            :param input: Input data or output from the previous layer
            :return: Layer output
        """
        # store the input for backpropagation
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward_step(self, grad_output):
        """
            Back propagate the gradient through the layer
            :param grad_output: The gradient of the loss w.r.t the output of the layer
                passed doesn from the subsequent layer (or from the loss function if this is the final layer)
                indicates how the loss would change with small changes in the output of this layer
            :return: The gradient of the loss w.r.t the input of the layer
        """
        # calculates the gradient of the loss function w.r.t the layers weights
        # derived using the chain rule, combining the gradient of the loss w.r.t the output of the layer (grad_output) and the layers input (self.input)
        # transpose of input is necessary for mat mult to match dimensions
        # essentially shows how changes in weights would affect the loss
        grad_weights = np.dot(self.input.T, grad_output)

        # calculates the gradient of the loss function w.r.t the layers biases
        # since biases are added directly to the weighted inputs (i.e. deriv is 1), their gradients are just the sum of the gradients passed to the layer summed across the batch
        # axis=0 sums across rows (batch dimension) meaning we are summing across the batch dimension
            # effectively collapsing 'batch_size' number of gradients into a single gradient (num_batches -> 1)
        # keepdims=True ensures that the output has the same dimensions as the input 
        # summing gradients across all examples in the batch gives you a total gradient for each bias that reflects its influence on the loss across the entire batch
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)

        # calculates the gradient of the loss function w.r.t the input of the layer
        # this is needed for back propagation through previous layers
        # uses current grad output adn transposed weights to distribute the gradient through the weights, back to the inputs
        grad_input = np.dot(grad_output, self.weights.T)

        self.grad_weights = grad_weights
        self.grad_biases = grad_biases
        return grad_input
    
    def get_parameters_and_gradients(self):
        """
            Get the parameters of the layer and their gradients
            :return: A list of tuples where each tuple contains a reference to a parameter (weight or bias) and its corresponding gradient
        """
        return [(self.weights, self.grad_weights), (self.biases, self.grad_biases)]
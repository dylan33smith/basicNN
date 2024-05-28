from abc import ABC, abstractmethod
import numpy as np

class Optimizer(ABC):

    def __init__(self, parameters, learning_rate=0.01):
        """
            Constructor for the Optimizer class
            :param parameters: list of tuples where each tuple contains a reference to a parameter(weight matrix or bias vector) and its corresponding gradient
        """
        self.parameters = parameters
        self.learning_rate = learning_rate

    # each optimizer will need to override the update method to adjust network parameters
    @abstractmethod
    def update(self):
        """
            Update the parameters of the model based on the gradients computed during back propagation
        """
        pass

class SGD(Optimizer):

    def __init__(self, parameters, learning_rate=0.01):
        """
            Constructor for the SGD class
        """
        # extend init method with optimizer-specific parameters and store them as attributes
        super().__init__(parameters, learning_rate)

    def update(self):
        """
            Update the parameters of the model based on the gradients computed during back propagation
        """
        # iterate through each parameter and update it based on the gradient and learning rate
        for param, grad in self.parameters:
            param -= self.learning_rate * grad
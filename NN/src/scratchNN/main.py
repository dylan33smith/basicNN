import pandas as pd
import numpy as np
# from matplotlib import pyplot as plt

# import data is in the form of a tuple containing the training data and the testing data



# have class for:
    # tensor class
        # handles basic operations like add, mult, dot, and gradients
        # foundation for all computations
    # layer class/classes
        # define base class for layers ('Layer')
        # specific layer types (dense, convolutional, etc) will inherit
    # activation function class
        # implement activation functions as classes (Relu, sigmoid, etc) 
        # include both forward and backward methods
    # Loss function class
        # base class for loss function ('Loss')
        # subclasses for specific implementations (MSE, Crossentropy, etc)
    # Optimizer class
        # optimizers adjust the weights of NN based on gradients computed during backprop
        # basics (SGD, Adam) ensure they interact seamlessly with network parameters
    # Model class
        # class that ties everything together
            # hold a collection of layers
            # perform forward and backward passes
            # use loss function to calculate error
            # update model weights using an optimizer

# Key hyperparameters
    # Learning rate (optimizer)
        # affects steps taken during weight updates
    # batch size (Model)
        # influences memory efficiency and training stability
    # epoch # (Model or as separate trianing control function)
        # how many times the training loop runs through the entire dataset
    # layer dimension (Laer)
        # size and number of neurons in each layer
        # crucial for network capacity
    # Activation functions (integrated into layer or separate class)
        # impacts training dynamics and model performance


# Design
    # modular
        # ensure each class and function has single responsibility and easily interchangeable with others
    # Unit Testing
        # validate class/function behavior independently
    # Work flow
        # enable chaining operations to make model construction more intuitive
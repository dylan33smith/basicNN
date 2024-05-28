import numpy as np

class Simple_Loss:
    # Loss functions compute the difference between the predicted output and the true output
    # Loss functions are used to train the network by adjusting the network parameters to minimize the loss
    
    def __init__(self):
        pass
    
    def get_network_loss(self, y_pred, y_true):
        """
            Compute the loss
            :param y_pred: The predicted output of the network
            :param y_true: The true output of the network
            :return: The loss
        """
        # should I have something to ensure that y_pred and y_true are the same shape?
        return y_true - y_pred
    
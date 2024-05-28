import pandas as pd
import numpy as np
import os


def load_training_data(train_data_path):
    ''' Take in train csv file and return a tuple
    containing the shuffled images and labels as 
    numpy arrays.

        
    
    Parameters:
        train_data_path (str): The path to the training data csv file
        test_data_path (str): The path to the testing data csv file
    
    Returns:
        tuple: A tuple containing the training data and the testing data
    
        The first entry contains the training images as an ndarray
        with shape (num_images, num_pixels) where num_images is the
        number of images in the data set and num_pixels is 784
        (28x28 pixels).

        The second entry contains the labels for the images as an
        ndarray with shape (num_images, 1) where num_images is the
        number of images in the data set. Each entry in the dataset
        contains the digit value for the corresponding image
    '''

    # data contains a ndarray where the first col is the label
    # and the rest of the cols are the pixel values
    train_data = pd.read_csv(train_data_path)
    train_data = train_data.sample(frac=.01).reset_index(drop=True)
    train_input_data = np.array(train_data.iloc[:, 1:].values)

    # normalize the pixel values
    train_input_data = train_input_data / 255.0
    # transpose image data so that each column is an observation (image)
    train_input_data = train_input_data

    train_labels = np.array(train_data.iloc[:, 0].values)
    train_labels = one_hot_encode_label(train_labels)

    return (train_input_data, train_labels)


def load_testing_data(test_data_path):
    ''' Take in test csv file and return an ndarray of the 
        unlabeled test images

    
    Parameters:
        test_data_path (str): The path to the testing data csv file
    
    Returns:
        ndaray: An ndarray containing the testing images
    '''
    # data contains a ndarray where the first col is the label
    # and the rest of the cols are the pixel values
    test_data = pd.read_csv(test_data_path)
    test_data = test_data.sample(frac=1).reset_index(drop=True)
    test_data = test_data.T
    
    return test_data


def one_hot_encode_label(labels):
    """Takes in a ndarray of number labels and returns a one-hot encoded    
    unit vector with a 1.0 in the  

    Parameters: a numpy array of integer labels

    returns: a 2D numpy array where each row corresponds to a one-hot encoded 
    representation of the label

    for non integer labels, labels will have to be mapped to integers
    for this method to work properly 
    - create a function for this??    
    """
    labels = labels.astype(int)
    num_labels = len(np.unique(labels))
    n = labels.shape[0] # number of observations/labels
    
    # initialize matrix of zeros
    oh_y = np.zeros((n, num_labels))

    for index, y in enumerate(labels):
        oh_y[index, y] = 1.0

    # transpose so that each col is an observation
    return oh_y

    
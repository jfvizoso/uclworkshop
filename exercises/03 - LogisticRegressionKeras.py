'''
A very simple example of logistic regression via Keras this time, used for the
Deep Learning workshop.

We will use the MNIST dataset of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Authors: Pablo Doval (@PabloDoval)
'''
import sys
sys.path.append('../') 
import keras as k
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.callbacks import TensorBoard
import numpy as np
from utils.init import init

if __name__ == '__main__':
    init()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Set a few variables to use later
    num_train = X_train.shape[0]
    height = X_train.shape[1]
    width = X_train.shape[2]
    num_test = X_test.shape[0]
    num_classes = 10
    training_epochs = 20
    batch_size = 50

    # Prepare data: 
    # Reshape to image format
    # Convert to float
    # Normalize
    # Convert to category

    # Build model (using the softmax activation function)

    # Compile model with desired loss function, optimizer and metrics

    # tensorboard callback

    # Train the model

    # Evaluate it against the test dataset

    # evaluation result includes both loss and accuracy
    print('Train loss: ', score[0])
    print('Train accuracy: ', score[1])

'''
A very simple example of logistic regression via TensorFlow, used for the Deep Learning workshop.
We will use the MNIST dataset of handwritten digits (http://yann.lecun.com/exdb/mnist/)

Authors: Pablo Doval (@PabloDoval)
'''
import sys
sys.path.append('../') 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils.init import MNIST_DATA_FOLDER, init

if __name__ == '__main__':
    init()
    
    # Download MNIST data if not loaded before
    mnist = input_data.read_data_sets(MNIST_DATA_FOLDER, one_hot=True)

    # Hyper Parameters setup
    learning_rate = 0.01
    training_epochs = 20
    batch_size = 50
    display_step = 1
    total_batch =  int(mnist.train.num_examples/batch_size)

    # Input nodes and Weights (x, y, W, b)

    # This is the model: Just a basic Logistic Regression (Softmax of xW+b)
    

    # Definition of our cost function (cross entropy)
    

    # Set up of the optimizer (Gradient Descent)
    

    # Global Variables Initializer
    
    # Training process, using TF session.
        # Initialize Global Variables 

        # for each epoch
            # reset average cost            

            # for each batch
                # read batch and train
                       
                # update average cost                                     
            
            # print on each display_step


        print("Training finished.")

        # Test the trained model
        # Calculate correct_prediction comparing prediction with y, remember 1-hot encoding

        # Compute the model's accuracy, as mean of correct_prediction (1-hot encoding)
        # mean expects floats
        

        # Eval accuracy with test images.
        
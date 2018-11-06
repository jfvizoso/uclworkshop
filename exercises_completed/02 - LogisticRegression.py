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
    total_batch = int(mnist.train.num_examples/batch_size)

    # Input nodes and Weights (x, y, W, b)
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # This is the model: Just a basic Logistic Regression (Softmax of xW+b)
    pred = tf.nn.softmax(tf.matmul(x, W) + b) 

    # Definition of our cost function (cross entropy)
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred)))
    
    # Set up of the optimizer (Gradient Descent)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    init = tf.global_variables_initializer()

    # Training process, using TF session.
    with tf.Session() as sess:
        # Initialize Global Variables
        sess.run(init)

        # for each epoch
        for epoch in range(training_epochs):
            # reset average cost
            avg_cost = 0.
            

            # for each batch
            for i in range(total_batch):
                # read batch and train
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_xs, y: batch_ys})
                       
                # update average cost                                     
                avg_cost += c / total_batch
            
            # print on each display_step
            if (epoch+1) % display_step == 0:
                print("Epoch:", (epoch+1), "Loss=", "{:.9f}".format(avg_cost))

        print("Training finished.")

        # Test the trained model
        # Calculate correct_prediction comparing prediction with y, remember 1-hot encoding
        correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

        # Compute the model's accuracy, as mean of correct_prediction (1-hot encoding)
        # mean expects floats
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Eval accuracy with test images.
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
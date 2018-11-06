'''
A TensorFlow 'hello world', performing a very basic operation, to explain 
the concepts of the session, operators, etc.

Authors: Pablo Doval (@PabloDoval)
'''
import sys
sys.path.append('../') 
import tensorflow as tf
from utils.init import init, TF_LOGGING_FOLDER

if __name__ == '__main__':
    init()
   
    # --- CONSTANT OPERATORS ---
    # Step 1: Let's define a couple of constant operators here
    op1 = tf.constant(10)
    op2 = tf.constant(20)

    # Step 2: Create a session where to perform addition and multiplication of
    # the ops
    with tf.Session() as sess:
        print("Sum of constants ", sess.run(op1 + op2))
        print("Multiplication of constants ", sess.run(op1 * op2))
    
   
    # --- PLACEHOLDERS---
    # Step 1: Define two int16 placeholders so we can operate with them later
    pl1 = tf.placeholder('float')
    pl2 = tf.placeholder('float')

    # Step 2: Define our addition and multiplication operations
    add = tf.add(pl1, pl2)
    sum = tf.multiply(pl1, pl2)

    # Step 3: Create a session for these operations
    with tf.Session() as sess:
        print(sess.run(add, feed_dict={pl1: 10, pl2: 30}))

    # -- MATRIXES ---
    # Step 1: Define two constant matrixes. The first one, 1x2, while the 
    # second is 2x1
    m1 = [[1, 2, 3]]
    m2 = [[1],[2],[3]]


    # Step 2: Define the matrix multiplication
    product = tf.matmul(m1,m2)

    # Step 3: Create a session for these operations
    with tf.Session() as sess:
        sess.run(product)

    # --- DEBUG ---
    # Step 1: Create a session for the previous multiplication, but writing the
    # output to the tensorflow log.
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(TF_LOGGING_FOLDER)

        sess.run(product)

        writer.close()
    # Step 2: Analyze the graph with TensorBoard


    
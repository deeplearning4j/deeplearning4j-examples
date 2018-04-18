""" Multilayer Perceptron training on MNIST

Original code from:
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/


with small additions made to:
    1) save the model and a set of input+predictions
    2) name input and output ops
    3) separate out a softmax only op
"""

from __future__ import print_function
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Please change this to a local path
save_dir = "/Users/susaneraly/Desktop/saved_model"
# Parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# Network Parameters
n_hidden_1 = 256  # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input], name="input")
Y = tf.placeholder("float", [None, n_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer


def save_content(np_array, var_name):
    content_file = "{}/{}.csv".format(save_dir, var_name)
    shape_file = "{}/{}.shape".format(save_dir, var_name)
    np.savetxt(shape_file, np.asarray(np_array.shape), fmt="%i")
    np.savetxt(content_file, np.ndarray.flatten(np_array), fmt="%10.8f")


# Construct model
logits = multilayer_perceptron(X)
pred = tf.nn.softmax(logits, name="output")  # Apply softmax to logits

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()
all_saver = tf.train.Saver()

if "susaneraly" in save_dir:
    raise ValueError("Please change save_dir to a local path")

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                            Y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost={:.9f}".format(avg_cost))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

    # Additional code to save checkpoint and graph def
    all_saver.save(sess, "{}/{}".format(save_dir, "data-all"), global_step=1000)
    tf.train.write_graph(sess.graph_def, save_dir, "model.txt", True)

    # Saving off input and predictions to feed into nd4j/samediff also and check to see if the output is the same
    input_a = mnist.test.images[0].reshape(-1, 784);
    save_content(input_a, "input_a")
    prediction_a = pred.eval({X: input_a})
    save_content(prediction_a, "prediction_a")

    input_b = mnist.test.images[1].reshape(-1, 784);
    save_content(input_b, "input_b")
    prediction_b = pred.eval({X: input_b})
    save_content(prediction_b, "prediction_b")


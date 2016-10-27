"""
Author:
    Ximing Qiao, qiaoximing@gmail.com

"""

import os.path
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tsne3

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10


def get_error_rate(logits, labels):
    """Get percentage error rate

    Args:
        logits: tensor of predicted logits
        labels: tensor of correct labels

    Returns:
        tensor of percentage of error rate

    """
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    error_rate = 100 * (1 - tf.reduce_mean(tf.cast(correct_prediction, "float")))
    return error_rate


def teacher_model(data, labels):
    """Definition of the teacher network

    Args:
        data: tensor of input images
        labels: tensor of input labels

    Returns:
        train_step: train step of the model
        loss: tensor of training loss
        logits: tensor of predicted logits
        hidden: tensor of last hidden layer

    """
    conv1_weights = tf.Variable(tf.truncated_normal(
        [5, 5, NUM_CHANNELS, 32], stddev=0.1), name="t_conv1_w")
    conv1_biases = tf.Variable(tf.zeros([32]), name="t_conv1_b")

    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, 32, 64], stddev=0.1), name="t_conv2_w")
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]), name="t_conv2_b")

    fc1_weights = tf.Variable(tf.truncated_normal(
        [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev=0.1),
        name="t_fc1_w")
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]), name="t_fc1_b")

    fc2_weights = tf.Variable(tf.truncated_normal(
        [512, NUM_LABELS], stddev=0.1), name="t_fc2_w")
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]),
                             name="t_fc2_b")

    data = tf.reshape(data, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

    conv = tf.nn.conv2d(data, conv1_weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    conv = tf.nn.conv2d(pool, conv2_weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))
    pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

    reshape = tf.reshape(pool, [-1, IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64])

    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    logits = tf.matmul(hidden, fc2_weights) + fc2_biases

    # dropout when training
    T = 1
    hidden1 = tf.nn.dropout(hidden, 0.5, seed=1)
    logits1 = tf.matmul(hidden1, fc2_weights) + fc2_biases
    correct1 = tf.cast(tf.equal(tf.argmax(logits1, 1), 
                                tf.argmax(labels, 1)), "float")
    soft1 = tf.exp(logits1 / T) / tf.reduce_sum(tf.exp(logits1 / T))
    hidden2 = tf.nn.dropout(hidden, 0.5, seed=100)
    logits2 = tf.matmul(hidden2, fc2_weights) + fc2_biases
    correct2 = tf.cast(tf.equal(tf.argmax(logits2, 1), 
                                tf.argmax(labels, 1)), "float")
    soft2 = tf.exp(logits2 / T) / tf.reduce_sum(tf.exp(logits2 / T))

    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    #     logits_drop, labels))
    gamma = 0
    loss = (gamma * correct1 * (1 - correct2) * (-tf.reduce_sum(tf.stop_gradient(soft1) * tf.log(soft2))) + 
        gamma * (1 - correct1) * correct2 * (-tf.reduce_sum(tf.stop_gradient(soft2) * tf.log(soft1))) + 
        (-tf.reduce_sum(labels * tf.log(soft1)) - tf.reduce_sum(labels * tf.log(soft2))))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return train_step, loss, logits, hidden


def train_teacher(sess, data, labels, train_step, loss, error):
    """Train teacher network

    Args:
        sess: tensorflow session
        data: tensor of input image
        labels: tensor of input labels
        train_step: training step of teacher network
        loss: tensor of loss of teacher network
        error: tensor of error rate of teacher network

    """
    sess.run(tf.initialize_all_variables())

    for i in range(10000):
        # train on batches
        batch = mnist.train.next_batch(50)
        _, train_loss, train_error = sess.run(
            [train_step, loss, error],
            feed_dict={data:batch[0], labels:batch[1]})

        # validate each 100 steps
        if i % 100 == 0:
            val_error = sess.run( error, feed_dict={
                data:mnist.validation.images,
                labels:mnist.validation.labels})
            print("step %d, loss %.3f, train error %.1f, val error %.1f"
                  % (i, train_loss, train_error, val_error))


def main():
    """

    """
    # set placeholders for input data
    data = tf.placeholder("float", shape=[
        None, IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS])
    labels = tf.placeholder("float", shape=[None, NUM_LABELS])

    sess = tf.Session()

    # train or load the teacher network
    train_step, loss, logits, hidden = teacher_model(data, labels)
    error = get_error_rate(logits, labels)
    train_teacher(sess, data, labels, train_step, loss, error)

    sess.close()


if __name__ == "__main__":
    main()

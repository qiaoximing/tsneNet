"""
Implementation of neural network training with t-SNE regularization.

Note:
    MNIST dataset will be downloaded if not found.
    teacher.ckpt and P_batch.npy will be created for caching in the first run.

Author:
    Ximing Qiao, qiaoximing@gmail.com

"""

import os.path
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tsne3

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

# Training data size and batch size for the student network
TRAIN_SIZE = 6000
BATCH_SIZE = 100


def calcP(X, initial_dims, perplexity):
    """Calculate the similarity matrix P

    Reduce the dimension of X to initial_dims using PCA,
    then calculate P by binary search.

    Args:
        X: distance matrix X
        initial_dims: initial dimension for PCA
        perplexity: user specified parameter

    Returns:
        similarity matrix P

    """
    X = tsne3.pca(X, initial_dims).real
    P = tsne3.x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = np.maximum(P, 1e-12)
    return P


def get_tsne_loss(P, Y, alpha):
    """Tensorflow operation for calculating tsne-loss

    Args:
        P: tensor of similarity matrix P
        Y: tensor of last-hidden-layer of student network
        alpha: dimension of student-t distribution,
               set to infinity if alpha <= 0

    Returns:
        tensor of tsne-loss

    """
    sum_Y = tf.reduce_sum(tf.square(Y), 1)
    num = tf.transpose(-2 * tf.matmul(Y, tf.transpose(Y)) + sum_Y) + sum_Y
    if alpha > 0:
        num = tf.pow(1 + num / alpha, -(alpha + 1) / 2)
    else:
        num = tf.exp(-num / 2)
    noeye = 1 - tf.diag(tf.ones([BATCH_SIZE]))
    num *= noeye
    Q = num / tf.reduce_sum(num)
    Q = tf.maximum(Q, 1e-12)
    C = tf.reduce_sum(P * tf.log(P / Q))
    return C


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
    hidden_drop = tf.nn.dropout(hidden, 0.5)
    logits_drop = tf.matmul(hidden_drop, fc2_weights) + fc2_biases

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits_drop, labels))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    return train_step, loss, logits, hidden


def train_or_load_teacher(sess, mnist, data, labels, train_step, loss, error):
    """Train or load teacher network

    Train a teacher network and save to ./teacher.ckpy,
    load if existed.

    Args:
        sess: tensorflow session
        data: tensor of input image
        labels: tensor of input labels
        train_step: training step of teacher network
        loss: tensor of loss of teacher network
        error: tensor of error rate of teacher network

    """
    filepath = "./teacher.ckpt"

    if not os.path.isfile(filepath):
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

        # save model
        saver = tf.train.Saver()
        saver.save(sess, filepath)
        print("Model saved in file: %s" % filepath)

    else:
        # load if existed
        saver = tf.train.Saver()
        saver.restore(sess, filepath)
        print("Model loaded in file: %s" % filepath)


def student_model(data, labels, node_P, dropout=True, alpha=-1, beta=0):
    """Definition of the student network

    Args:
        data: tensor of input images
        labels: tensor of input labels
        node_P: tensor of similarity matirx P
        dropout: Ture if use dropout in training
        alpha: parameter alpha in tnse-loss
        beta: parameter beta in hybrid loss

    Returns:
        train_step: training step of the model
        loss: tensor of training cross-entropy loss
        tsne_loss: tensor of training tsne-loss
        logits: tensor of predicted logits

    """
    size1 = 8
    size2 = 16
    size3 = 32

    conv1_weights = tf.Variable(tf.truncated_normal(
        [5, 5, NUM_CHANNELS, size1], stddev=0.1), name="s_conv1_w")
    conv1_biases = tf.Variable(tf.zeros([size1]), name="s_conv1_b")

    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, size1, size2], stddev=0.1), name="s_conv2_w")
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[size2]),
                               name="s_conv2_b")

    fc1_weights = tf.Variable(tf.truncated_normal(
        [IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * size2, size3], stddev=0.1),
        name="s_fc1_w")
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[size3]), name="s_fc1_b")

    fc2_weights = tf.Variable(tf.truncated_normal(
        [size3, NUM_LABELS], stddev=0.1), name="s_fc2_w")
    fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS]),
                             name="s_fc2_b")

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

    reshape = tf.reshape(pool, [-1, IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * size2])

    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
    logits = tf.matmul(hidden, fc2_weights) + fc2_biases

    if dropout:
        hidden_drop = tf.nn.dropout(hidden, 0.5)
        logits_drop = tf.matmul(hidden_drop, fc2_weights) + fc2_biases
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits_drop, labels))
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits, labels))

    tsne_loss = get_tsne_loss(node_P, hidden, alpha)

    train_step = tf.train.AdamOptimizer(5e-4).minimize(loss + beta * tsne_loss)

    return train_step, loss, tsne_loss, logits


def main():
    """Full training procedure of student network

    In the first run, this program download the MNIST dataset,
    trains the teacher network,
    and caches P matrices for all batches.
    After that, it starts to train the student network.

    """
    # load mnist data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # set placeholders for input data
    data = tf.placeholder("float", shape=[
        None, IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS])
    labels = tf.placeholder("float", shape=[None, NUM_LABELS])
    node_P = tf.placeholder("float", shape=[BATCH_SIZE, BATCH_SIZE])

    # calculate or load P matrices in batches
    P_batch = []
    filepath = "./P_batch.npy"

    if not os.path.isfile(filepath):
        sess = tf.Session()

        # train or load the teacher network
        train_step, loss, logits, hidden = teacher_model(data, labels)
        error = get_error_rate(logits, labels)
        train_or_load_teacher(sess, mnist, data, labels, train_step, loss,
                              error)

        # calc hidden-layer-value of training data, using teacher network
        teacher_hidden = sess.run(
            hidden, feed_dict={data:mnist.train.images[:TRAIN_SIZE]})

        # calc a list of P matrices, each on one batch of hidden-layer-value
        P_batch = []
        for i in range(0, TRAIN_SIZE - BATCH_SIZE + 1, BATCH_SIZE):
            P_batch.append(calcP(teacher_hidden[i:i + BATCH_SIZE],
                                 initial_dims=50, perplexity=20))
        np.save(filepath, np.array(P_batch))
        print("P_batch saved")

        sess.close()

    else:
        # load if existed
        P_batch = np.load(filepath).tolist()
        print("P_batch loaded")

    sess = tf.Session()

    # set student network
    train_step, loss, tsne_loss, logits = student_model(
        data, labels, node_P, dropout=True, alpha=-1, beta=0.2)
    error = get_error_rate(logits, labels)

    # start training
    sess.run(tf.initialize_all_variables())

    for i in range(10000):
        offset = i % (TRAIN_SIZE // BATCH_SIZE)
        _, train_loss, train_tsne_loss, train_error = sess.run(
            [train_step, loss, tsne_loss, error], feed_dict={
            data:mnist.train.images[
                offset * BATCH_SIZE:(offset + 1) * BATCH_SIZE],
            labels:mnist.train.labels[
                offset * BATCH_SIZE:(offset + 1) * BATCH_SIZE],
            node_P:P_batch[offset]})

        if i % 100 == 0:
            val_error = sess.run(error, feed_dict={
                data:mnist.validation.images,
                labels:mnist.validation.labels})
            print("step %d, loss %.3f, tsne loss %.2f, train error %.2f, val error %.2f"
                  % (i, train_loss, train_tsne_loss, train_error, val_error))


if __name__ == "__main__":
    main()

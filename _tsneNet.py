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

IMAGE_SIZE = 56
NUM_CHANNELS = 3
NUM_LABELS = 3

# Training data size and batch size for the student network
TRAIN_SIZE = 3000
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
    size1 = 16
    size2 = 16
    size3 = 16

    conv1_weights = tf.Variable(tf.truncated_normal(
        [5, 5, NUM_CHANNELS, size1], stddev=0.01), name="s_conv1_w")
    conv1_biases = tf.Variable(tf.zeros([size1]), name="s_conv1_b")

    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, size1, size2], stddev=0.01), name="s_conv2_w")
    conv2_biases = tf.Variable(tf.constant(0.01, shape=[size2]),
                               name="s_conv2_b")

    fc1_weights = tf.Variable(tf.truncated_normal(
        [(IMAGE_SIZE-8) * (IMAGE_SIZE-8) * size2, size3], stddev=0.01),
        name="s_fc1_w")
    fc1_biases = tf.Variable(tf.constant(0.01, shape=[size3]), name="s_fc1_b")

    fc2_weights = tf.Variable(tf.truncated_normal(
        #[size3, NUM_LABELS], stddev=0.01), name="s_fc2_w")
        [(IMAGE_SIZE-8) * (IMAGE_SIZE-8) * size2, NUM_LABELS], stddev=0.01), name="s_fc2_w")
    fc2_biases = tf.Variable(tf.constant(0.01, shape=[NUM_LABELS]),
                             name="s_fc2_b")

    conv = tf.nn.conv2d(data, conv1_weights,
                        strides=[1, 1, 1, 1], padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    conv = tf.nn.conv2d(relu, conv2_weights,
                        strides=[1, 1, 1, 1], padding='VALID')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

    hidden = tf.reshape(relu, [-1, (IMAGE_SIZE-8) * (IMAGE_SIZE-8) * size2])

    #hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
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
    #train_step = tf.train.AdamOptimizer(5e-3).minimize(loss)

    return train_step, loss, tsne_loss, logits


def main():
    """Full training procedure of student network

    In the first run, this program download the MNIST dataset,
    trains the teacher network,
    and caches P matrices for all batches.
    After that, it starts to train the student network.

    """
    # load image data
    cache_dir='image_retraining/'
    mnist = {'train':{}, 'validation':{}, 'test':{}}
    mnist['train']['images'] = np.load(cache_dir+'train.images.npy')/255
    mnist['train']['bottlenecks'] = np.load(cache_dir+'train.bottlenecks.npy')
    mnist['train']['labels'] = np.load(cache_dir+'train.labels.npy')
    mnist['validation']['images'] = np.load(cache_dir+'validation.images.npy')/255
    mnist['validation']['bottlenecks'] = np.load(cache_dir+'validation.bottlenecks.npy')
    mnist['validation']['labels'] = np.load(cache_dir+'validation.labels.npy')
    print(mnist['train']['images'].shape)

    # set placeholders for input data
    data = tf.placeholder("float", shape=[
        None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    labels = tf.placeholder("float", shape=[None, NUM_LABELS])
    node_P = tf.placeholder("float", shape=[BATCH_SIZE, BATCH_SIZE])

    # calculate or load P matrices in batches
    P_batch = []
    filepath = "./_P_batch.npy"

    if not os.path.isfile(filepath):

        teacher_hidden = mnist['train']['bottlenecks'][:TRAIN_SIZE]

        # calc a list of P matrices, each on one batch of hidden-layer-value
        P_batch = []
        for i in range(0, TRAIN_SIZE - BATCH_SIZE + 1, BATCH_SIZE):
            P_batch.append(calcP(teacher_hidden[i:i + BATCH_SIZE],
                                 initial_dims=50, perplexity=20))
        np.save(filepath, np.array(P_batch))
        print("P_batch saved")

    else:
        # load if existed
        P_batch = np.load(filepath).tolist()
        print("P_batch loaded")

    sess = tf.Session()

    # set student network
    train_step, loss, tsne_loss, logits = student_model(
        data, labels, node_P, dropout=False, alpha=-1, beta=0)
    error = get_error_rate(logits, labels)

    # start training
    sess.run(tf.initialize_all_variables())

    for i in range(10000):
        offset = i % (TRAIN_SIZE // BATCH_SIZE)
        _, train_loss, train_tsne_loss, train_error = sess.run(
            [train_step, loss, tsne_loss, error], feed_dict={
            data:mnist['train']['images'][
                offset * BATCH_SIZE:(offset + 1) * BATCH_SIZE],
            labels:mnist['train']['labels'][
                offset * BATCH_SIZE:(offset + 1) * BATCH_SIZE],
            node_P:P_batch[offset]})

        if i % 100 == 0:
            val_error = sess.run(error, feed_dict={
                data:mnist['validation']['images'],
                labels:mnist['validation']['labels']})
            print("step %d, loss %.3f, tsne loss %.2f, train error %.2f, val error %.2f"
                  % (i, train_loss, train_tsne_loss, train_error, val_error))
            #p = np.random.permutation(len(mnist['train']['images']))
            #mnist['train']['images'] = mnist['train']['images'][p]
            #mnist['train']['bottlenecks'] = mnist['train']['bottlenecks'][p]
            #mnist['train']['labels'] = mnist['train']['labels'][p]


if __name__ == "__main__":
    main()

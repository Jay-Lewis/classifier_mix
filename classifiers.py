import numpy as np
import tensorflow as tf
from tensorlayer.layers import InputLayer, Conv2dLayer, MaxPool2d, LocalResponseNormLayer, FlattenLayer, DenseLayer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import PIL.Image

import utils


def celebA_classifier(ims, reuse):
    with tf.variable_scope("C", reuse=reuse) as vs:
        net = InputLayer(ims)
        n_filters = 3
        for i in range(2):
            net = Conv2dLayer(net, \
                    act=tf.nn.relu, \
                    shape=[5,5,n_filters,64], \
                    name="conv_" + str(i))
            net = MaxPool2d(net, \
                    filter_size=(3,3), \
                    strides=(2,2), \
                    name="mpool_" + str(i))
            net = LocalResponseNormLayer(net, \
                    depth_radius=4, \
                    bias=1.0, \
                    alpha=0.001 / 9.0, \
                    beta=0.75, \
                    name="lrn_" + str(i))
            n_filters = 64
        net = FlattenLayer(net)
        net = DenseLayer(net, n_units=384, act=tf.nn.relu, name="d1")
        net = DenseLayer(net, n_units=192, act=tf.nn.relu, name="d2")
        net = DenseLayer(net, n_units=2, act=tf.identity, name="final")
        cla_vars = tf.contrib.framework.get_variables(vs)
        if not reuse:
            return net.outputs, tf.argmax(net.outputs, axis=1), cla_vars
    return net.outputs, tf.argmax(net.outputs, axis=1)

def cifar10_classifier(im, reuse):
    with tf.variable_scope('C', reuse=reuse) as vs:
        net = InputLayer(im)
        net = Conv2dLayer(net, \
                act=tf.nn.relu, \
                shape=[5,5,3,64], \
                name="conv1")
        net = MaxPool2d(net, \
                filter_size=(3,3), \
                strides=(2,2), \
                name="pool1")
        net = LocalResponseNormLayer(net, \
                depth_radius=4, \
                bias=1.0, \
                alpha = 0.001/9.0, \
                beta = 0.75, \
                name="norm1")
        net = Conv2dLayer(net, \
                act=tf.nn.relu, \
                shape=[5,5,64,64], \
                name="conv2")
        net = LocalResponseNormLayer(net, \
                depth_radius=4, \
                bias=1.0, \
                alpha=0.001/9.0, \
                beta = 0.75, \
                name="norm2")
        net = MaxPool2d(net, \
                filter_size=(3,3), \
                strides=(2,2), \
                name="pool2")
        net = FlattenLayer(net, name="flatten_1")
        net = DenseLayer(net, n_units=384, name="local3", act=tf.nn.relu)
        net = DenseLayer(net, n_units=192, name="local4", act=tf.nn.relu)
        net = DenseLayer(net, n_units=10, name="softmax_linear", act=tf.identity)
        cla_vars = tf.contrib.framework.get_variables(vs)
        def name_fixer(var):
            return var.op.name.replace("W", "weights") \
                                .replace("b", "biases") \
                                .replace("weights_conv2d", "weights") \
                                .replace("biases_conv2d", "biases")
        cla_vars = {name_fixer(var): var for var in cla_vars}
        if not reuse:
            return net.outputs, tf.argmax(net.outputs, axis=1), cla_vars
        return net.outputs, tf.argmax(net.outputs, axis=1)

def dnn_model_fn(x):
    dense1 = tf.layers.dense(inputs=x, units=10, activation=tf.nn.relu)
    dense2 = tf.layers.dense(inputs=dense1, units=20, activation=tf.nn.relu)
    dense_out = tf.layers.dense(inputs=dense2, units=10, activation=tf.nn.relu)
    return tf.layers.dense(inputs=dense_out, units=2)

def cnn_model_fn(x, dropout_rate, reuse=False):
    with tf.variable_scope("Classifier", reuse=reuse):
        xn = x + tf.random_normal(tf.shape(x), dropout_rate/2, 0.001 + dropout_rate, dtype=tf.float32)
        xn = tf.clip_by_value(xn, 0, 1)
        
        input_layer = tf.reshape(xn, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate)

        outputs = tf.layers.dense(inputs=dropout, units=10)
        classifier_params = [var for var in tf.trainable_variables() if 'Classifier' in var.name]

        return outputs, classifier_params

def cnn_model_fn_mixed(x, dropout_rate, reuse=False):
    with tf.variable_scope("Classifier", reuse=reuse):
        xn = x + tf.random_normal(tf.shape(x), dropout_rate / 2, 0.001 + dropout_rate, dtype=tf.float32)
        xn = tf.clip_by_value(xn, 0, 1)

        input_layer = tf.reshape(xn, [-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate)

        outputs = tf.layers.dense(inputs=dropout, units=10)
        classifier_params = [var for var in tf.trainable_variables() if 'Classifier' in var.name]

        return outputs, classifier_params

def save_model(sess, saver, checkpoint_dir):
    saver.save(sess, checkpoint_dir + 'trained_model')
    saver.export_meta_graph(checkpoint_dir + 'trained_model_graph' + '.meta')

def eval_swiss_model(sess, x, y, logits, x_test, y_test, accuracy_op, image_path=None, name=None):
    print(name, "Accuracy:", sess.run(accuracy_op, feed_dict={x: x_test, y: y_test}))
    if image_path:
        test_pred = sess.run(tf.argmax(logits, 1), feed_dict={x: x_test, y: y_test})
        test_pred_c = np.array(['b' if pred == 0 else 'r' for pred in test_pred])
        plt.scatter([x_test[:, 0]], [x_test[:, 1]], c=test_pred_c)
        plt.savefig(image_path)
        plt.figure()

        
def make_one_hot(coll):
    onehot = np.zeros((coll.shape[0], coll.max() + 1))
    onehot[np.arange(coll.shape[0]), coll] = 1
    return onehot

def restore_model(sess, saver, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt: saver.restore(sess, ckpt.model_checkpoint_path)

def get_accuracy_op(logits, y):
    correct_pred = tf.equal(tf.argmax(logits, 1),
                            tf.argmax(y, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def get_train_op(logits, y, learning_rate):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    loss_op = tf.losses.softmax_cross_entropy(y, logits=logits)
    return optimizer.minimize(loss_op)

def train_model(sess, x, y, train_epoch, train_op, num_epochs, batch_size):
    train_gen = utils.batch_gen(train_epoch, True, y.shape[1], num_epochs)
    for x_train, y_train in train_gen:
        sess.run(train_op, feed_dict={x: x_train, y: y_train})

def get_batches(colls, batch_size):
    return apply(zip, [np.array_split(coll, batch_size) for coll in colls])

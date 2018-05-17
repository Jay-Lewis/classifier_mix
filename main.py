
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import pickle
import scipy

from classifiers import *


def main():

    # ----------------------------------------------------------------
    # Create Dataset (if not already)
    # -----------------------------------

    filepath = './data/mixed_mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print('Mixed MNIST dataset not found at:')
        print(filepath)
        print('Creating mixed MNIST dataset...')
        utils.create_MNIST_mixed()

    #-----------------------------------------------------------------
    # Set up Classifier + training parameters
    #-----------------------------------------
    train_classifier = True
    normal_MNIST = True

    learning_rate = 0.001
    num_epochs = 2
    batch_size = 100
    x_size = 784
    y_size = 10
    dropout_rate = tf.placeholder_with_default(0.4, shape=())
    x = tf.placeholder(tf.float32, shape=[None, x_size])
    y = tf.placeholder(tf.int32, shape=[None, y_size])

    # Load Training Data + Define Model
    if(normal_MNIST):
        load_func = utils.MNIST_load
        classifier_model_fn = lambda x: cnn_model_fn(x, dropout_rate)
        logits, classifier_params = classifier_model_fn(x)
        classifier_saver = tf.train.Saver(var_list=classifier_params, max_to_keep=1)
        classifier_model_directory = "./model_Classifier/MNIST/"
    else:
        load_func = utils.MNIST_load_mixed()
        classifier_model_fn = lambda x: cnn_model_fn_mixed(x, dropout_rate)
        logits, classifier_params = classifier_model_fn(x)
        classifier_saver = tf.train.Saver(var_list=classifier_params, max_to_keep=1)
        classifier_model_directory = "./model_Classifier/MNIST_mixed/"

    # Define Training Operation
    train_op = get_train_op(logits, y, learning_rate)

    # Initialization
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    #-----------------------------------------------------------------
    # Train Classifier
    #-----------------------------------------

    with tf.Session(config = config) as sess:
        sess.run(init)

        if train_classifier:
            print ('Start Training Classifier')
            train_epoch, _, test_epoch = utils.load_dataset(batch_size, load_func)
            print('training model')
            train_model(sess, x, y, train_epoch, train_op, num_epochs, batch_size)
            print('saving model')
            save_model(sess,classifier_saver,classifier_model_directory)
        else:
            print('Load Classifier')
            restore_model(sess,classifier_saver,classifier_model_directory)
            print('Loaded from:')
            print(classifier_model_directory)


        #--------------------------------------------------
        # Test Classifier
        #----------------------------
        accuracy_op = get_accuracy_op(logits, y)
        batch_size = 100
        train_epoch, _,test_epoch = utils.load_dataset(batch_size, load_func)
        batch_gen = utils.batch_gen(test_epoch, True, y.shape[1], num_iter=1)
        iteration = 0
        normal_avr = 0
        for images, labels in batch_gen:
            iteration += 1
            avr = sess.run(accuracy_op, feed_dict={x: images, y: labels, dropout_rate: 0.0})
            normal_avr += avr
        print("Normal Accuracy:", normal_avr / iteration)



if __name__ == '__main__': main()

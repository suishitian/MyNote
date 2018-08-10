import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

INPUTNODE = 784
OUTPUT_NODE = 10

FC_SIZE = 512

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

CONV1_DEEP = 32
CONV1_SIZE = 5

CONV2_DEEP = 64
CONV2_SIZE = 5

TRAIN_STEPS = 10000
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DACAY = 0.99
REGU_RATE = 0.0001
BATCH_SIZE = 100


def inference(input_tensor,regu,trainable):
    print("inference")
    print(input_tensor.get_shape())
    with tf.variable_scope('layer1_conv1'):
        conv1_weights = tf.get_variable("l1_weights",[CONV1_SIZE,CONV1_SIZE,1,CONV1_DEEP],initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("l1_biases",[CONV1_DEEP],initializer = tf.constant_initializer(0.1))

        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='SAME')
        conv1_output = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))

    with tf.variable_scope("layer2_pool"):
        pool1_output = tf.nn.max_pool(conv1_output,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    with tf.variable_scope("layer3_conv2"):
        conv2_weights = tf.get_variable("l3_weights",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer = tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("l3_biases",[CONV2_DEEP],initializer = tf.constant_initializer(0.1))
        conv2 = tf.nn.conv2d(pool1_output,conv2_weights,strides=[1,1,1,1],padding='SAME')
        conv2_output = tf.nn.relu(tf.nn.bias_add(conv2,conv2_biases))

    with tf.variable_scope("layer4_pool2"):
        pool2_output = tf.nn.max_pool(conv2_output,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
    ##pull the matrix to a vertor with one dimension
    pool_shape = pool2_output.get_shape().as_list()
    nodes = pool_shape[1]*pool_shape[2]*pool_shape[3]
    print(pool_shape)
    reshaped = tf.reshape(pool2_output,[BATCH_SIZE,nodes])

    with tf.variable_scope("layer4_fc1"):
        fc1_weights = tf.get_variable("l4_weights",[nodes,FC_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1_biases = tf.get_variable("l4_biases",[FC_SIZE],initializer=tf.constant_initializer(0.1))
        if regu:
            tf.add_to_collection("losses",regu(fc1_weights))
        fc1_output = tf.nn.relu(tf.matmul(reshaped,fc1_weights)+fc1_biases)
        if trainable:
            fc1_output = tf.nn.dropout(fc1_output,0.5)

    with tf.variable_scope("layer5_fc2"):
        fc2_weights = tf.get_variable("l5_weights",[FC_SIZE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable("l5_biases",[OUTPUT_NODE],initializer=tf.constant_initializer(0.1))
        if regu:
            tf.add_to_collection("losses",regu(fc2_weights))
        output = tf.matmul(fc1_output,fc2_weights)+fc2_biases
    return output
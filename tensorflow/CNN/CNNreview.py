import tensorflow as tf
import numpy as np
import inference
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

def train(mnist):
    y_ph = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name="input_y")
    x_ph = tf.placeholder(tf.float32,[None,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS],name="input_x")

    regu = tf.contrib.layers.l2_regularizer(REGU_RATE)
    y_pre = inference.inference(x_ph,regu,True)

    global_step = tf.Variable(0,trainable=False)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pre,labels=tf.argmax(y_ph,1))
    loss = tf.reduce_mean(cross_entropy)+tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DACAY,
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    acc_m = tf.equal(tf.argmax(y_pre, 1), tf.argmax(y_ph, 1))
    acc = tf.reduce_mean(tf.cast(acc_m, tf.float32))

    with tf.Session() as sess:
        print("start training")
        sess.run(tf.global_variables_initializer())

        x_test_feed = mnist.validation.images
        y_test_feed = mnist.validation.labels

        for i in range(TRAIN_STEPS):
            print("step:",i)
            x_feed, y_feed = mnist.train.next_batch(BATCH_SIZE)
            x_feed_reshaped = np.reshape(x_feed, (
                BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS,
            ))
            print("before train")
            sess.run(train_step,feed_dict={x_ph:x_feed_reshaped,y_ph:y_feed})
            if i%100==0:
                x_test_feed_reshaped = np.reshape(x_test_feed, (
                    mnist.validation.images.shape[0],
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    NUM_CHANNELS,
                ))
                loss_pre,acc_pre = sess.run([loss,acc],feed_dict={x_ph:x_test_feed_reshaped,y_ph:y_test_feed})
                print(i,"step: loss:",loss_pre,"  acc:  ",acc_pre)
        loss_final,acc_final = sess.run([loss,acc],feed_dict={x_ph:x_test_feed,y_ph:y_test_feed})
        print("final:  loss: ",loss_final,"  acc:   ",acc_final)

def main(argv=None):
    mnist = input_data.read_data_sets("../MNIST_data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()


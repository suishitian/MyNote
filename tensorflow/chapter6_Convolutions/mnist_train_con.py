import tensorflow as tf
import os
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference_con

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.5
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAIN_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "/data"
MODEL_NAME = "model.ckpt"
def train(mnist):
    print "train start"
    #x = tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x-input")
    y_ = tf.placeholder(tf.float32,[None,mnist_inference_con.OUTPUT_NODE],name="y-input")
    x = tf.placeholder(tf.float32,[
        BATCH_SIZE,
        mnist_inference_con.IMAGE_SIZE,
        mnist_inference_con.IMAGE_SIZE,
        mnist_inference_con.NUM_CHANNELS],
                       name="x-input")

    
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    y = mnist_inference_con.inference(x,True,regularizer)
    global_step = tf.Variable(0,trainable=False)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op = ema.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step,ema_op]):
        train_op = tf.no_op(name='train')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print "before for loop"
        for i in range(TRAIN_STEPS):
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs,(BATCH_SIZE,
                                 mnist_inference_con.IMAGE_SIZE,
                                 mnist_inference_con.IMAGE_SIZE,
                                 mnist_inference_con.NUM_CHANNELS))

            _,loss_value,step = sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
            
            if i%100==0:
                print("After %d training step, loss is %g"%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step = global_step)
    #print "over"
def main(argv=None):
    mnist=input_data.read_data_sets("./MNIST_data/",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()














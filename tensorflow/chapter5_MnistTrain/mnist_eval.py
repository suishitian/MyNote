import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import mnist_inference
import mnist_train

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x_input")
        y_=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],name="y_input")
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}

        y=mnist_inference.inference(x,None)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        ema = tf.train.ExponentialMovingAverage(mnist_train.MOVING_AVERAGE_DECAY)
        variable_restore = ema.variables_to_restore()
        saver = tf.train.Saver(variable_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    acc_score = sess.run(acc,feed_dict=validate_feed)
                    print("After %s training step,validation acc = %g"%(global_step,acc_score))
                else:
                    print("No checkpoint file")
                    return
                time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data",one_hot = True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()

                    















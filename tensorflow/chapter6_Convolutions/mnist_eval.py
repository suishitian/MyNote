import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_inference_con
import mnist_train_con

EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        #x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],name="x_input")
        x = tf.placeholder(tf.float32,[
        	None,
        	mnist_inference_con.IMAGE_SIZE,
        	mnist_inference_con.IMAGE_SIZE,
        	mnist_inference_con.NUM_CHANNELS],
                       name="x-input")
        print("eval start")
        y_=tf.placeholder(tf.float32,[None,mnist_inference_con.OUTPUT_NODE],name="y_input")
        #print(mnist.validation.images[0:100])
        #xs,ys = mnist.train.next_batch(100)
        #sample_shape = mnist.validation.images.get_shape().as_list()
        
        reshaped_xs = np.reshape(mnist.validation.images,(mnist.validation.images.shape[0],
                                 mnist_inference_con.IMAGE_SIZE,
                                 mnist_inference_con.IMAGE_SIZE,
                                 mnist_inference_con.NUM_CHANNELS))
        #print("label")
        print(reshaped_xs.shape)
        #validate_feed = {x:reshaped_xs,y_:mnist.validation.labels}

        y=mnist_inference_con.inference(x,False,None)
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        ema = tf.train.ExponentialMovingAverage(mnist_train_con.MOVING_AVERAGE_DECAY)
        variable_restore = ema.variables_to_restore()
        saver = tf.train.Saver(variable_restore)

        while True:
            with tf.Session() as sess:
                print("eval start")
                ckpt = tf.train.get_checkpoint_state(mnist_train_con.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)            

                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    acc_score = sess.run(acc,feed_dict={x:reshaped_xs,y_:mnist.validation.labels})
                    print("After %s training step,validation acc = %g"%(global_step,acc_score))
                else:
                    print("No checkpoint file")
                    return
                time.sleep(EVAL_INTERVAL_SECS)
def main(argv=None):
    mnist=input_data.read_data_sets("./MNIST_data/",one_hot = True)
    evaluate(mnist)
if __name__ == '__main__':
    tf.app.run()

                    















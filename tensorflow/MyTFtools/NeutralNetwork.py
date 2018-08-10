import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
INPUT_SIZE = 784
OUTPUT_SIZE = 10
LAYER1_SIZE = 500
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGU = 0.0001
TRAIN_STEP = 10000
EMA_RATE = 0.99
BATCH_SIZE = 100

def inference(input_tensor, regu,reuse):
    with tf.variable_scope('layer1',reuse = reuse):
        weights = tf.get_variable("weights",shape=[INPUT_SIZE,LAYER1_SIZE],initializer = tf.truncated_normal_initializer(stddev=0.1))
        if regu!=None:
            tf.add_to_collection('losses',regu(weights))
        bias = tf.get_variable("bias",shape=[LAYER1_SIZE],initializer = tf.constant_initializer(0.1))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+bias)
    with tf.variable_scope('layer2',reuse = reuse):
        weights = tf.get_variable("weight",shape=[LAYER1_SIZE,OUTPUT_SIZE],initializer = tf.truncated_normal_initializer(stddev=0.1))
        if regu!=None:
            tf.add_to_collection('losses',regu(weights))
        bias = tf.get_variable("bias",shape=[OUTPUT_SIZE],initializer = tf.constant_initializer(0.1))
        layer2 = tf.matmul(layer1,weights)+bias
    return layer2

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_SIZE],name="x-input")
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_SIZE],name="y-input")

    regu = tf.contrib.layers.l2_regularizer(REGU)
    _ = inference(x,regu,False)
    y = inference(x,regu,True)

    global_step = tf.Variable(0,trainable=False)

    #ema = tf.train.ExponentialMovingAverage(EMA_RATE,global_step)
    #ema_op = ema.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples/BATCH_SIZE,
        LEARNING_RATE_DECAY
    )

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)

    acc = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accu = tf.reduce_mean(tf.cast(acc,tf.float32))

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        validate_feed = {x:mnist.validation.images, y_:mnist.validation.labels}
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}

        for i in range(TRAIN_STEP):
            if i%100 == 0:
                validate_acc = sess.run(accu,feed_dict = validate_feed)
                print(i," steps acc: ",validate_acc)
            xs,ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step,feed_dict={x:xs,y_:ys})
        test_acc = sess.run(accu,feed_dict=test_feed)
        print("the final acc is ",test_acc)

def main(atgv=None):
    mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

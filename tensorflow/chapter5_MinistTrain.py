import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#
INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER_NODE = 500

BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
#
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2
    else:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weights1))+avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+avg_class.average(biases2)
#
def train(mnist):
	#placeholder
	x = tf.placeholder(tf.float32,[None,INPUT_NODE],name="x-input")
	y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name="y-input")

	weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER_NODE],stddev=0.1))
	biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER_NODE]))

	weights2 = tf.Variable(tf.truncated_normal([LAYER_NODE,OUTPUT_NODE],stddev=0.1))
	biases2 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

	global_step = tf.Variable(0,trainable=False)
	#
	y = inference(x,None,weights1,biases1,weights2,biases2)
	#
	variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
	variables_averages_op = variable_average.apply(tf.trainable_variables())
	#
	average_y = inference(x,variable_average,weights1,biases1,weights2,biases2)
	#
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = y,labels = tf.argmax(y_,1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)

	#
	regu = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	regu_total = regu(weights1) + regu(weights2)
	#
	loss = cross_entropy_mean+regu_total

	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,
		global_step,
		mnist.train.num_examples/BATCH_SIZE,
		LEARNING_RATE_DECAY)
#
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
		global_step=global_step)
	with tf.control_dependencies([train_step,variables_averages_op]):
		train_op = tf.no_op(name='train')

	correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
	correct_prediction_noa = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuary = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	accuary_noa = tf.reduce_mean(tf.cast(correct_prediction_noa,tf.float32))
	
	with tf.Session() as sess :
		tf.initialize_all_variables().run()
		validate_feed = {x:mnist.validation.images,
					y_:mnist.validation.labels}
		test_feed = {x:mnist.test.images, y_:mnist.test.labels}
		

		for i in range(TRAINING_STEPS):
			if i%1000==0:
				validate_acc= sess.run(accuary,feed_dict = validate_feed)
				print("After %d training steps, validation accuracy using average model is %g"
					%(i,validate_acc))
				##print sess.run(weights1);
			xs,ys = mnist.train.next_batch(BATCH_SIZE)
			sess.run(train_op,feed_dict={x:xs,y_:ys})
		
		test_acc = sess.run(accuary,feed_dict=test_feed)
		print("After %d training steps, test accuracy using average model is %g"
				%(TRAINING_STEPS,test_acc))
	print "over"
def main(atgv = None):
	mnist = input_data.read_data_sets("tmp/data",one_hot=True)
	train(mnist)

if __name__ == '__main__':
	tf.app.run();
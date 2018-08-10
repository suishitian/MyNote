import tensorflow as tf
v = tf.Variable(0,dtype = tf.float32,name="v")

for variable in tf.all_variables():
    print variable.name
    
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.all_variables())
saver = tf.train.Saver()
for variable in tf.all_variables():
    print variable.name
    
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    sess.run(tf.assign(v,10))
    sess.run(maintain_averages_op)

    saver.save(sess,"model1.ckpt")
    print sess.run([v,ema.average(v)])

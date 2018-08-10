import tensorflow as tf

files = tf.train.match_filenames_once("data.tfrecords-*")

filename_queue = tf.train.string_input_producer(files,shuffle=False)

reader = tf.TFRecordReader()
_,Serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    Serialized_example,
    features={
        'i':tf.FixedLenFeature([],tf.int64),
        'j':tf.FixedLenFeature([],tf.int64),
    }
)

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    print sess.run(files)
    ##print files

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    #for i in range(6):
    #    print sess.run([features['i'],features['j']])
    #coord.request_stop()
    #coord.join(threads)

    example , label = features['i'],features['j']

    batch_size = 3

    capacity = 1000+3*batch_size

    example_batch,label_batch = tf.train.batch(
        [example,label],batch_size=batch_size,capacity=capacity
    )


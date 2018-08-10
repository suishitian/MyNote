import tensorflow as tf

files = tf.train.match_filenames_once("data.tfrecord-*")
filename_queue = tf.train.string_input_producer(files,shuffle=False)

reader = tf.TFRecordReader()
_, Serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    Serialized_example,
    features={
        ''
    }
)
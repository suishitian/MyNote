import tensorflow as tf
import matplotlib.pyplot as plt

image_raw_data = tf.gfile.FastGFile("test.jpg","r").read()

with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)

    img_data = tf.image.convert_image_dtype(img_data, dtype=tf.float32)

    resized = tf.image.resize_images(img_data,[800,800],method=3)

    resized_u = tf.image.convert_image_dtype(resized, dtype=tf.uint8)

    ##croped = tf.image.resize_image_with_crop_or_pad(img_data,200,200)
    ##croped = tf.image.central_crop(img_data,0.5)
    ##flipped=tf.image.random_flip_up_down(img_data)
    ##max_delta = 2
    ##adjust = tf.image.per_image_standardization(img_data)
    img_data = tf.image.resize_images(img_data,[200,200],method=1)
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data,tf.float32),0)
    boxed = tf.constant([[[0.05,0.05,0.9,0.9],[0.35,0.47,0.5,0.56]]])

    result = tf.image.draw_bounding_boxes(img_data,boxes=boxed)
    target = tf.image.convert_image_dtype(result[0], dtype=tf.uint8)
    encoded_image = tf.image.encode_jpeg(target)
    filename = "croped.jpg"
    with tf.gfile.GFile(filename,"wb") as f:
        print("write over")
        f.write(encoded_image.eval())

    plt.imshow(tf.image.decode_jpeg(tf.gfile.FastGFile(filename,"rb").read()).eval())
    plt.show()

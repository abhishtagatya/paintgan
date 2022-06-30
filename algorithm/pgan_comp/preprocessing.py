import tensorflow as tf


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    # Map values in the range [-1, 1]
    return (img / 127.5) - 1.0


def preprocess_train_image(img, image_size=(256, 256, 3)):

    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)
    # Random flip
    img = tf.image.random_flip_left_right(img)
    # Resize to the original size first
    img = tf.image.resize(img, [image_size[0], image_size[1]])
    # Random crop to 256X256
    # Normalize the pixel values in the range [-1, 1]
    img = normalize_img(img)
    return img


def preprocess_test_image(img, image_size=(256, 256, 3)):

    img = tf.io.read_file(img)
    img = tf.image.decode_jpeg(img, channels=3)

    # Only resizing and normalization for the test images.
    img = tf.image.resize(img, [image_size[0], image_size[1]])
    img = normalize_img(img)
    return img
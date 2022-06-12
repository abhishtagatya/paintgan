import tensorflow as tf


def preprocess_image(image_path, size=(256, 256)):
    """
      Preprocess the image by decoding and resizing and image
      from the image file path.

      Args:
        image_path: The image file path.
        size: The size of the image to be resized to.

      Returns:
        image: resized image
    """

    max_dim = size[0]

    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    shape = tf.cast(tf.shape(image)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    image = tf.image.resize(image, new_shape)
    image = image[tf.newaxis, :]
    return image

import tensorflow as tf


def decode_and_resize(image_path, size):
    """
        Decodes and resizes and image from the image file path.

        Args:
          image_path: The image file path.
          size: The size of the image to be resized to.

        Returns:
          image: resized image
      """

    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype='float32')
    image = tf.image.resize(image, size)

    return image

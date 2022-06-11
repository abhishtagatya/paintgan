import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


class ReflectionPadding2D(tf.keras.layers.Layer):
    """
      Implements Reflection Padding as a layer

      Args:
        padding(tuple): Amount of padding for the spatial dimensions

      Returns:
        A padded tensor with the same type as the input tensor
    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0]
        ]
        return tf.pad(input_tensor, padding_tensor, mode='REFLECT')


def residual_block(
        x,
        activation,
        kernel_initializer,
        gamma_initializer,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=False
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D().call(input_tensor)
    x = tf.keras.layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D().call(x)
    x = tf.keras.layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = tf.keras.layers.add([input_tensor, x])

    return x


def downsample(
        x,
        filters,
        activation,
        kernel_initializer,
        gamma_initializer,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False
):
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def upsample(
        x,
        filters,
        activation,
        kernel_initializer,
        gamma_initializer,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        use_bias=False
):
    x = tf.keras.layers.Conv2DTranspose(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=kernel_initializer,
        use_bias=use_bias
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    return x


def get_resnet_generator(
        image_size,
        kernel_initializer,
        gamma_initializer,
        filters=64,
        num_downsampling_blocks=2,
        num_residual_blocks=9,
        num_upsample_blocks=2,
        name=None
):
    img_input = tf.keras.layers.Input(shape=image_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3)).call(img_input)
    x = tf.keras.layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = tf.keras.layers.Activation("relu")(x)

    # Downsample
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(
            x,
            filters=filters,
            activation=tf.keras.layers.Activation("relu"),
            kernel_initializer=kernel_initializer,
            gamma_initializer=gamma_initializer
        )

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(
            x,
            activation=tf.keras.layers.Activation("relu"),
            kernel_initializer=kernel_initializer,
            gamma_initializer=gamma_initializer
        )

    # Upsample
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(
            x,
            filters,
            activation=tf.keras.layers.Activation("relu"),
            kernel_initializer=kernel_initializer,
            gamma_initializer=gamma_initializer
        )

    # Final block
    x = ReflectionPadding2D(padding=(3, 3)).call(x)
    x = tf.keras.layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = tf.keras.layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


def get_discriminator(
        image_size,
        kernel_initializer,
        gamma_initializer,
        filters=64,
        num_downsampling=3,
        name=None
):
    img_input = tf.keras.layers.Input(shape=image_size, name=name + "_img_input")
    x = tf.keras.layers.Conv2D(
        filters,
        (4, 4),
        strides=(2, 2),
        padding="same",
        kernel_initializer=kernel_initializer
    )(img_input)
    x = tf.keras.layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(
                x,
                filters=num_filters,
                activation=tf.keras.layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(2, 2),
                kernel_initializer=kernel_initializer,
                gamma_initializer=gamma_initializer,
            )
        else:
            x = downsample(
                x,
                filters=num_filters,
                activation=keras.layers.LeakyReLU(0.2),
                kernel_size=(4, 4),
                strides=(1, 1),
                kernel_initializer=kernel_initializer,
                gamma_initializer=gamma_initializer,
            )

    x = tf.keras.layers.Conv2D(
        1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer
    )(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
    return model


def generator_loss_fn(fake, loss_fn=keras.losses.MeanSquaredError()):
    fake_loss = loss_fn(tf.ones_like(fake), fake)
    return fake_loss


def discriminator_loss_fn(real, fake, loss_fn=keras.losses.MeanSquaredError()):
    real_loss = loss_fn(tf.ones_like(real), real)
    fake_loss = loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5

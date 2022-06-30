import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


def unet_conv2d(layer_input, filters, f_size=4, normalize=True):
    d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, padding='same')(layer_input)
    d = tf.keras.layers.LeakyReLU(0.2)(d)
    if normalize:
        d = tfa.layers.InstanceNormalization()(d)
    return d


def unet_deconv2d(layer_input, skip, filters, f_size=4, dropout=0):
    u = tf.keras.layers.UpSampling2D(size=2)(layer_input)
    u = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
    if dropout:
        u = tf.keras.layers.Dropout(dropout)(u)
    u = tfa.layers.InstanceNormalization()(u)
    u = tf.keras.layers.Concatenate()([
        u, skip
    ])
    return u


def get_unet_generator(
        image_size=(256, 256, 3),
        filters=64
):
    h, w, c = image_size

    # Image input
    d0 = tf.keras.layers.Input(shape=image_size)

    # Downsample
    d1 = unet_conv2d(d0, filters, normalize=False)
    d2 = unet_conv2d(d1, filters * 2)
    d3 = unet_conv2d(d2, filters * 4)
    d4 = unet_conv2d(d3, filters * 8)
    d5 = unet_conv2d(d4, filters * 8)
    d6 = unet_conv2d(d5, filters * 8)
    d7 = unet_conv2d(d6, filters * 8)

    # Upsample
    u1 = unet_deconv2d(d7, d6, filters * 8)
    u2 = unet_deconv2d(u1, d5, filters * 8)
    u3 = unet_deconv2d(u2, d4, filters * 8)
    u4 = unet_deconv2d(u3, d3, filters * 4)
    u5 = unet_deconv2d(u4, d2, filters * 2)
    u6 = unet_deconv2d(u5, d1, filters)

    u7 = tf.keras.layers.UpSampling2D(size=2)(u6)
    output_img = tf.keras.layers.Conv2D(
        c, kernel_size=4, strides=1,
        padding='same', activation='tanh'
    )(u7)

    return tf.keras.models.Model(
        d0, output_img
    )


def discriminator_layer(layer_input, filters, f_size=4, normalization=True):
    d = tf.keras.layers.Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = tf.keras.layers.LeakyReLU(0.2)(d)
    if normalization:
        d = tfa.layers.InstanceNormalization()(d)
    return d


def get_discriminator(
        image_size=(256, 256, 3),
        filters=64
):
    h, w, c = image_size

    # Image input
    d0 = tf.keras.layers.Input(shape=image_size)

    # Downsampling
    d1 = discriminator_layer(d0, filters, normalization=False)
    d2 = discriminator_layer(d1, filters * 2)
    d3 = discriminator_layer(d2, filters * 4)
    d4 = discriminator_layer(d3, filters * 8)

    # Using PatchGAN Discriminator Method
    val_d = tf.keras.layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return tf.keras.models.Model(d0, val_d)


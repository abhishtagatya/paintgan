import tensorflow as tf
from keras import Model, Sequential
from keras.layers import Input, InputLayer, Conv2D, UpSampling2D
from keras.applications import VGG19


def get_encoder(image_size):
    """
    AdaIN Encoder using VGG16 Pretrained on ImageNet Weights

    :return: tf.keras.Model
    """
    vgg19 = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3)
    )
    vgg19.trainable = False
    mini_vgg19 = Model(vgg19.input, vgg19.get_layer("block4_conv1").output)

    inputs = Input([*image_size, 3])
    mini_vgg19_out = mini_vgg19(inputs)

    return Model(
        inputs,
        mini_vgg19_out,
        name="mini_vgg19"
    )


def get_mean_std(x, epsilon=1e-5):
    axes = [1, 2]

    # Compute the mean adn standard deviation of a tensor
    mean, var = tf.nn.moments(x, axes=axes, keepdims=True)
    std_dev = tf.sqrt(var + epsilon)

    return mean, std_dev


def ada_in_func(style, content):
    """
    Computes the AdaIN feature map

    :param style:
    :param content:
    :return:
    """

    content_mean, content_std = get_mean_std(content)
    style_mean, style_std = get_mean_std(style)

    t = style_std * (content - content_mean) / content_std + style_mean

    return t


def get_decoder(config=None):
    """
    Building Decoder Model in a Sequential Upsampling Layers

    :return: tf.keras.Sequential
    """

    if config is None:
        config = {
            "kernel_size": 3,
            "strides": 1,
            "padding": "same",
            "activation": "relu"
        }

    decoder = Sequential(
        [
            InputLayer((None, None, 512)),
            Conv2D(filters=512, **config),
            UpSampling2D(),
            Conv2D(filters=256, **config),
            Conv2D(filters=256, **config),
            Conv2D(filters=256, **config),
            Conv2D(filters=256, **config),
            UpSampling2D(),
            Conv2D(filters=128, **config),
            Conv2D(filters=128, **config),
            UpSampling2D(),
            Conv2D(filters=64, **config),
            Conv2D(
                filters=3,
                kernel_size=3,
                strides=1,
                padding="same",
                activation="sigmoid",
            ),
        ]
    )

    return decoder


def get_loss_net(image_size, layer_blocks=None):
    vgg19 = VGG19(
        include_top=False,
        weights="imagenet",
        input_shape=(*image_size, 3)
    )
    vgg19.trainable = False

    layer_names = layer_blocks
    if layer_names is None:
        layer_names = [
            "block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"
        ]

    outputs = [
        vgg19.get_layer(name).output for name in layer_names
    ]
    mini_vgg19 = Model(vgg19.input, outputs)

    inputs = Input([*image_size, 3])
    mini_vgg19_out = mini_vgg19(inputs)

    return Model(
        inputs,
        mini_vgg19_out,
        name="loss_net"
    )


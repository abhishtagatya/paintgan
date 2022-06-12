import numpy as np
import tensorflow as tf

from PIL import Image


def vgg_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    # Load our model. Load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_loss_fn(style_outputs, style_targets, num_style_layers, style_weight=1e-2):
    loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                     for name in style_outputs.keys()])
    loss *= style_weight / num_style_layers
    return loss


def content_loss_fn(content_outputs, content_targets, num_content_layers, content_weight=1e4):
    loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                     for name in content_outputs.keys()])
    loss *= content_weight / num_content_layers
    return loss


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

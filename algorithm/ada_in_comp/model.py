import tensorflow as tf
from tensorflow import keras

from algorithm.ada_in_comp.func import ada_in_func, get_mean_std


class AdaptiveInstanceNorm(tf.keras.Model):

    def __init__(self, encoder, decoder, loss_net, style_weight, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.loss_net = loss_net
        self.style_weight = style_weight

        self.optimizer = None
        self.loss_fn = None

        self.style_loss_tracker = keras.metrics.Mean(name="style_loss")
        self.content_loss_tracker = keras.metrics.Mean(name="content_loss")
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")

    def compile(self, optimizer, loss_fn):
        super().compile()
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def train_step(self, inputs):
        content, style = inputs

        # Initialize the content and style loss
        loss_content = 0.0
        loss_style = 0.0

        with tf.GradientTape() as tape:
            # Encode the style and content image
            style_encoded = self.encoder(style)
            content_encoded = self.encoder(content)

            # Compute the AdaIN target feature maps
            t = ada_in_func(style=style_encoded, content=content_encoded)

            # Generate the neural style transferred image
            recon_image = self.decoder(t)

            # Compute the loss
            recon_vgg_features = self.loss_net(recon_image)
            style_vgg_features = self.loss_net(style)
            loss_content = self.loss_fn(t, recon_vgg_features[-1])

            for inp, out in zip(style_vgg_features, recon_vgg_features):
                mean_inp, std_inp = get_mean_std(inp)
                mean_out, std_out = get_mean_std(out)

                loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(std_inp, std_out)

            loss_style = self.style_weight * loss_style
            total_loss = loss_content + loss_style

        # Compute gradients and optimize the decoder
        trainable_vars = self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the trackers
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result()
        }

    def test_step(self, inputs):
        content, style = inputs

        # Initialize the content and style loss
        loss_content = 0.0
        loss_style = 0.0

        # Encode the style and content
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        # Compute the AdaIN feature maps
        t = ada_in_func(style=style_encoded, content=content_encoded)

        # Generate the neural style transferred image
        recon_image = self.decoder(t)

        recon_vgg_features = self.loss_net(recon_image)
        style_vgg_features = self.loss_net(style)
        loss_content = self.loss_fn(t, recon_vgg_features[-1])

        for inp, out in zip(style_vgg_features, recon_vgg_features):
            mean_inp, std_inp = get_mean_std(inp)
            mean_out, std_out = get_mean_std(out)
            loss_style += self.loss_fn(mean_inp, mean_out) + self.loss_fn(
                std_inp, std_out
            )
        loss_style = self.style_weight * loss_style
        total_loss = loss_content + loss_style

        # Update the trackers
        self.style_loss_tracker.update_state(loss_style)
        self.content_loss_tracker.update_state(loss_content)
        self.total_loss_tracker.update_state(total_loss)

        return {
            "style_loss": self.style_loss_tracker.result(),
            "content_loss": self.content_loss_tracker.result(),
            "total_loss": self.total_loss_tracker.result()
        }

    def inference(self, content, style):
        style_encoded = self.encoder(style)
        content_encoded = self.encoder(content)

        t = ada_in_func(style=style_encoded, content=content_encoded)

        recon_image = self.decoder(t)
        return recon_image

    @property
    def metrics(self):
        return [
            self.style_loss_tracker,
            self.content_loss_tracker,
            self.total_loss_tracker
        ]

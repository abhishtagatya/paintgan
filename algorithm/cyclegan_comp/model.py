import numpy as np

import tensorflow as tf
from tensorflow import keras


class CycleGenAdvNet(tf.keras.Model):

    def __init__(
            self,
            generator_G,
            generator_F,
            discriminator_X,
            discriminator_Y,
            lambda_cycle=10.0,
            lambda_identity=0.5
    ):
        super(CycleGenAdvNet, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

        self.gen_G_optimizer = None
        self.gen_F_optimizer = None
        self.disc_X_optimizer = None
        self.disc_Y_optimizer = None
        self.generator_loss_fn = None
        self.discriminator_loss_fn = None

    def compile(
            self,
            gen_G_optimizer,
            gen_F_optimizer,
            disc_X_optimizer,
            disc_Y_optimizer,
            gen_loss_fn,
            disc_loss_fn,
            checkpoint_path='',
    ):
        super(CycleGenAdvNet, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
        self.identity_loss_fn = keras.losses.MeanAbsoluteError()

        self.model_checkpoint = tf.train.Checkpoint(
            gen_G=self.gen_G,
            gen_F=self.gen_F,
            disc_X=self.disc_X,
            disc_Y=self.disc_Y,
            gen_G_optimizer=self.gen_G_optimizer,
            gen_F_optimizer=self.gen_F_optimizer,
            disc_X_optimizer=self.disc_X_optimizer,
            disc_Y_optimizer=self.disc_Y_optimizer
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            self.model_checkpoint,
            checkpoint_path,
            max_to_keep=5
        )

    def train_step(self, batch_data):
        # x is content and y is style
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adversarial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:
            # Content to style
            fake_y = self.gen_G(real_x, training=True)
            # Style to content -> y2x
            fake_x = self.gen_F(real_y, training=True)

            # Cycle (Content to style to content): x -> y -> x
            cycled_x = self.gen_F(fake_y, training=True)
            # Cycle (Style to content to style): y -> x -> y
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adversarial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (
                    self.identity_loss_fn(real_y, same_y)
                    * self.lambda_cycle
                    * self.lambda_identity
            )
            id_loss_F = (
                    self.identity_loss_fn(real_x, same_x)
                    * self.lambda_cycle
                    * self.lambda_identity
            )

            # Total generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generator
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminator
        disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables)
        )
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables)
        )

        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables)
        )
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables)
        )

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss
        }

    def test_step(self, batch_data):
        # x is content and y is style
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different
        # kinds of losses for the generators and discriminators.
        # We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we
        #    we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adversarial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        # Content to style
        fake_y = self.gen_G(real_x, training=True)
        # Style to content -> y2x
        fake_x = self.gen_F(real_y, training=True)

        # Cycle (Content to style to content): x -> y -> x
        cycled_x = self.gen_F(fake_y, training=True)
        # Cycle (Style to content to style): y -> x -> y
        cycled_y = self.gen_G(fake_x, training=True)

        # Identity mapping
        same_x = self.gen_F(real_x, training=True)
        same_y = self.gen_G(real_y, training=True)

        # Discriminator output
        disc_real_x = self.disc_X(real_x, training=True)
        disc_fake_x = self.disc_X(fake_x, training=True)

        disc_real_y = self.disc_Y(real_y, training=True)
        disc_fake_y = self.disc_Y(fake_y, training=True)

        # Generator adversarial loss
        gen_G_loss = self.generator_loss_fn(disc_fake_y)
        gen_F_loss = self.generator_loss_fn(disc_fake_x)

        # Generator cycle loss
        cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
        cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

        # Generator identity loss
        id_loss_G = (
                self.identity_loss_fn(real_y, same_y)
                * self.lambda_cycle
                * self.lambda_identity
        )
        id_loss_F = (
                self.identity_loss_fn(real_x, same_x)
                * self.lambda_cycle
                * self.lambda_identity
        )

        # Total generator loss
        total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
        total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

        # Discriminator loss
        disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
        disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss
        }

    def restore_checkpoint(self, checkpoint_path):
        self.model_checkpoint.restore(checkpoint_path)

    def inference(self, content):
        pred = self.gen_G(content)[0].numpy()
        pred = (pred * 127 + 127.5).astype(np.uint8)

        return pred


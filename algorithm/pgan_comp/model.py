import numpy as np

import tensorflow as tf
from tensorflow import keras


class PGAN:

    def __init__(self,
                 generator_AB,
                 generator_BA,
                 discriminator_A,
                 discriminator_B,
                 image_size=(256, 256, 3)
                 ):
        self.image_size = image_size

        self.generator_AB = generator_AB
        self.generator_BA = generator_BA
        self.discriminator_A = discriminator_A
        self.discriminator_B = discriminator_B

    def compile(self, optimizer, checkpoint_path=''):
        mse_loss = keras.losses.MeanSquaredError()
        mae_loss = keras.losses.MeanAbsoluteError()

        self.discriminator_A.compile(
            loss=mse_loss, optimizer=optimizer, metrics=['acc']
        )
        self.discriminator_B.compile(
            loss=mse_loss, optimizer=optimizer, metrics=['acc']
        )

        img_A = tf.keras.layers.Input(shape=self.image_size)
        img_B = tf.keras.layers.Input(shape=self.image_size)

        fake_B = self.generator_AB(img_A)
        fake_A = self.generator_BA(img_B)

        recon_A = self.generator_BA(fake_B)
        recon_B = self.generator_AB(fake_A)

        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False

        valid_A = self.discriminator_A(fake_A)
        valid_B = self.discriminator_B(fake_B)

        self.combined = tf.keras.models.Model(
            inputs=[img_A, img_B],
            outputs=[
                valid_A, valid_B,
                fake_B, fake_A,
                recon_A, recon_B
            ]
        )
        self.combined.compile(
            loss=[
                mse_loss, mse_loss,
                mae_loss, mae_loss,
                mae_loss, mae_loss
            ],
            optimizer=optimizer
        )

        self.model_checkpoint = tf.train.Checkpoint(
            discriminator_A=self.discriminator_A,
            discriminator_B=self.discriminator_B,
            combined=self.combined,
            optimizer=optimizer
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            self.model_checkpoint,
            checkpoint_path,
            max_to_keep=5
        )

    def restore_checkpoint(self):
        if self.checkpoint_manager.latest_checkpoint:
            self.model_checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def inference(self, content):
        pred = self.generator_AB.predict(content)
        pred = (pred[0] * 127.5 + 127.5).astype(np.uint8)

        return pred

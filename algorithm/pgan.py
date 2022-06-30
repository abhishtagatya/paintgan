"""
    PaintGAN
    Paper : WIP

    PaintGAN : Generative Adversarial Network for Turning
    Photo into Painting by Famous Painter

    Tensorflow Original Implementation
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt

from algorithm.base import Algorithm

from algorithm.pgan_comp.preprocessing import preprocess_train_image, preprocess_test_image
from algorithm.pgan_comp.func import get_unet_generator, get_discriminator
from algorithm.pgan_comp.monitor import DisplayMonitor, CheckpointMonitor
from algorithm.pgan_comp.model import PGAN
from util.data_loader import DomainDataLoader


class PaintGAN(Algorithm):

    def __init__(self,
                 content_dir='',
                 style_dir='',
                 domain='',
                 epochs=1,
                 batch_size=8,
                 image_size=(256, 256, 3),
                 patch=None,
                 checkpoint=None,
                 mode='train'
                 ):
        super(PaintGAN, self).__init__(content_dir, style_dir, epochs, batch_size, image_size, mode)
        self._create_result_folder()

        self.style_domain = domain

        patch = int(self.image_size[0] / 2 ** 4)
        self.patch_disc = (patch, patch, 1)

        self.disc_A = get_discriminator()
        self.disc_B = get_discriminator()

        self.gen_AB = get_unet_generator()
        self.gen_BA = get_unet_generator()

        self.optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

        self.checkpoint_path = f'{self.model_name}/checkpoints/{self.style_domain}_{self.epochs}'
        if checkpoint:
            self.checkpoint_path = checkpoint

        if self.mode == 'train':
            self.data_loader = DomainDataLoader(
                content_path=self.content_dir,
                style_path=self.style_dir,
                style_domain=self.style_domain
            )
            self.train_ds, self.test_ds = self.data_loader.as_dataset(
                batch_size=self.batch_size,
                preprocess_func_train=preprocess_train_image,
                preprocess_func_test=preprocess_test_image
            )

        self.model = self.build_model()

        if checkpoint:
            self.model.restore_checkpoint()

    def build_model(self):

        model = PGAN(
            generator_AB=self.gen_AB,
            generator_BA=self.gen_BA,
            discriminator_A=self.disc_A,
            discriminator_B=self.disc_B,
            image_size=self.image_size
        )

        model.compile(
            optimizer=self.optimizer,
            checkpoint_path=self.checkpoint_path
        )

        self.D_Loss_metric = keras.metrics.Mean(name="D_Loss")
        self.G_loss_metric = keras.metrics.Mean(name="G_loss")

        return model

    def train(self, checkpoint_per=10):

        for epoch in range(self.epochs):

            dm = DisplayMonitor(self.model, self.model_name, self.style_domain, self.test_ds)
            cm = CheckpointMonitor(self.model, self.model_name, self.style_domain, 5)

            print(f'\nEpoch {epoch + 1}/{self.epochs}')
            pb_i = Progbar(self.train_ds.cardinality().numpy(), stateful_metrics=[
                'D_Loss', 'G_Loss'
            ])

            for batch_i, (images_A, images_B) in enumerate(self.train_ds):

                maxed = max(len(images_A), len(images_B))

                valid = np.ones((maxed,) + self.patch_disc)
                fake = np.zeros((maxed,) + self.patch_disc)

                # Translates images to opposite domain
                fake_B = self.model.generator_AB.predict(images_A[:maxed])
                fake_A = self.model.generator_BA.predict(images_B[:maxed])

                DA_loss_real = self.model.discriminator_A.train_on_batch(images_A, valid)
                DA_loss_fake = self.model.discriminator_A.train_on_batch(fake_A, fake)
                DA_loss = 0.5 * np.add(DA_loss_real, DA_loss_fake)

                DB_loss_real = self.model.discriminator_B.train_on_batch(images_B, valid)
                DB_loss_fake = self.model.discriminator_B.train_on_batch(fake_B, fake)
                DB_loss = 0.5 * np.add(DB_loss_real, DB_loss_fake)

                D_loss = 0.5 * np.add(DA_loss, DB_loss)

                G_loss = self.model.combined.train_on_batch(
                    [images_A, images_B],
                    [valid, valid,
                     images_B, images_A,
                     images_A, images_B]
                )

                self.D_Loss_metric.update_state(D_loss)
                self.G_loss_metric.update_state(G_loss)

                metric_values = [
                    ('D_Loss', self.D_Loss_metric.result()),
                    ('G_Loss', self.G_loss_metric.result())
                ]

                pb_i.add(1, values=metric_values)

            self.D_Loss_metric.reset_states()
            self.G_loss_metric.reset_states()

            dm.on_epoch_end(epoch)
            cm.on_epoch_end(epoch)

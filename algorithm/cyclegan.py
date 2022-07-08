"""
    CycleGAN : Cyclic Consistency GAN
    Paper : https://arxiv.org/abs/1703.10593

    Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

    Tensorflow Implementation

    Original Implementation : https://github.com/junyanz/CycleGAN (PyTorch)
    Reference : https://www.tensorflow.org/tutorials/generative/cyclegan
"""

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger, ModelCheckpoint

from algorithm.base import Algorithm

from algorithm.cyclegan_comp.preprocessing import preprocess_train_image, preprocess_test_image
from algorithm.cyclegan_comp.func import get_resnet_generator, get_discriminator, \
    generator_loss_fn, discriminator_loss_fn
from algorithm.cyclegan_comp.monitor import DisplayMonitor, CheckpointMonitor
from algorithm.cyclegan_comp.model import CycleGenAdvNet
from util.data_loader import DomainDataLoader


class CycleGAN(Algorithm):

    def __init__(self,
                 content_dir='',
                 style_dir='',
                 domain='',
                 epochs=1,
                 batch_size=1,
                 buffer_size=256,
                 image_size=(256, 256, 3),
                 max_set=0,
                 checkpoint=None,
                 mode='train'
                 ):
        super(CycleGAN, self).__init__(content_dir, style_dir, epochs, batch_size, image_size, mode)
        self._create_result_folder()

        self.style_domain = domain
        self.max_set = max_set

        self.buffer_size = buffer_size

        self.kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

        self.gen_G = get_resnet_generator(
            name='generator_G',
            image_size=self.image_size,
            kernel_initializer=self.kernel_init,
            gamma_initializer=self.gamma_init,
        )

        self.gen_F = get_resnet_generator(
            name='generator_F',
            image_size=self.image_size,
            kernel_initializer=self.kernel_init,
            gamma_initializer=self.gamma_init,
        )

        self.disc_X = get_discriminator(
            name='discriminator_X',
            image_size=self.image_size,
            kernel_initializer=self.kernel_init,
            gamma_initializer=self.gamma_init
        )

        self.disc_Y = get_discriminator(
            name='discriminator_Y',
            image_size=self.image_size,
            kernel_initializer=self.kernel_init,
            gamma_initializer=self.gamma_init
        )

        self.gen_G_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.gen_F_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.disc_X_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.disc_Y_optimizer = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)

        self.gen_loss_fn = generator_loss_fn
        self.disc_loss_fn = discriminator_loss_fn

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
                buffer_size=self.buffer_size,
                preprocess_func_train=preprocess_train_image,
                preprocess_func_test=preprocess_test_image,
                max_set=self.max_set
            )

        self.model = self.build_model()

        if checkpoint:
            self.model.restore_checkpoint()

        if self.mode == 'train':
            self.monitors = [
                DisplayMonitor(
                    self.model_name,
                    self.style_domain,
                    self.test_ds
                ),
                CheckpointMonitor(
                    self.model_name,
                    self.style_domain,
                    self.model
                ),
                CSVLogger(
                    f'{self.model_name}_{self.style_domain}-{self.epochs}.csv',
                    append=True,
                    separator=';'
                )
            ]

    def build_model(self) -> tf.keras.Model:
        model = CycleGenAdvNet(
            generator_G=self.gen_G,
            generator_F=self.gen_F,
            discriminator_X=self.disc_X,
            discriminator_Y=self.disc_Y,
        )

        model.compile(
            gen_G_optimizer=self.gen_G_optimizer,
            gen_F_optimizer=self.gen_F_optimizer,
            disc_X_optimizer=self.disc_X_optimizer,
            disc_Y_optimizer=self.disc_Y_optimizer,
            gen_loss_fn=self.gen_loss_fn,
            disc_loss_fn=self.disc_loss_fn,
            checkpoint_path=self.checkpoint_path
        )

        return model

    def train(self):
        history = self.model.fit(
            self.train_ds,
            validation_data=self.test_ds,
            epochs=self.epochs,
            callbacks=self.monitors
        )

        return history

    def evaluate(self, content, save_filename='img.jpg', size=(256, 256, 3)):
        content_image = preprocess_test_image(content)

        content_image = tf.reshape(
            tf.convert_to_tensor(
                content_image
            ),
            (1, *size)
        )
        recon_image = self.model.inference(content_image)

        if self.mode == 'inference':
            keras.preprocessing.image.save_img(f'{self.model_name}/inferences/{save_filename}', recon_image)

        if self.mode == 'evaluate':
            save_name = content.rsplit(".", 1)[0].split('/')[-1] + '_stylized_' + save_filename
            keras.preprocessing.image.save_img(f'{self.model_name}/evaluates/{save_name}', recon_image)

        keras.preprocessing.image.save_img(f'{save_filename}', recon_image)


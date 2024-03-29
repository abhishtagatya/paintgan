"""
    AdaIN : Adaptive Instance Normalization
    Paper : https://arxiv.org/pdf/1703.06868v2.pdf

    Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization

    Tensorflow Implementation

    Original Implementation : https://github.com/xunhuang1995/AdaIN-style (Lua)
    Reference : https://keras.io/examples/generative/adain/#downloading-the-dataset-from-kaggle
"""
import os.path

import tensorflow as tf
from tensorflow import keras
from keras.callbacks import CSVLogger, ModelCheckpoint

from algorithm.base import Algorithm

from algorithm.ada_in_comp.preprocessing import decode_and_resize
from algorithm.ada_in_comp.func import get_encoder, get_decoder, get_loss_net
from algorithm.ada_in_comp.monitor import DisplayMonitor, CheckpointMonitor
from algorithm.ada_in_comp.model import AdaptiveInstanceNorm
from util.data_loader import ArbitraryDataLoader


class AdaIN(Algorithm):

    def __init__(self,
                 content_dir='',
                 style_dir='',
                 epochs=1,
                 batch_size=32,
                 steps_per_epoch=100,
                 image_size=(256, 256),
                 style_weight=4.0,
                 checkpoint=None,
                 mode='train'
                 ):
        super(AdaIN, self).__init__(content_dir, style_dir, epochs, batch_size, image_size, mode)
        self._create_result_folder()
        self.steps_per_epoch = steps_per_epoch

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.encoder = get_encoder(self.image_size)
        self.loss_net = get_loss_net(self.image_size)
        self.decoder = get_decoder()

        self.style_weight = style_weight

        if self.mode == 'train':
            self.data_loader = ArbitraryDataLoader(content_path=self.content_dir, style_path=self.style_dir)
            self.train_ds, self.test_ds = self.data_loader.as_dataset(
                preprocess_func=decode_and_resize,
                batch_size=self.batch_size
            )

        self.model = self.build_model()

        if checkpoint:
            self.model.load_weights(checkpoint)

        if self.mode == 'train':
            self.monitors = [
                DisplayMonitor(
                    self.model_name,
                    self.test_ds
                ),
                ModelCheckpoint(
                    filepath=f'{self.model_name}/checkpoints/{self.model_name}_{self.epochs}.ckpt',
                    save_weights_only=True,
                    monitor='val_total_loss',
                    mode='min',
                    save_best_only=True,
                    save_freq='epoch'
                ),
                CSVLogger(
                    f'{self.model_name}/logs/{self.model_name}_{self.epochs}-{self.batch_size}.csv',
                    append=True,
                    separator=';'
                )
            ]

    def build_model(self) -> tf.keras.Model:
        model = AdaptiveInstanceNorm(
            encoder=self.encoder,
            decoder=self.decoder,
            loss_net=self.loss_net,
            style_weight=self.style_weight
        )

        model.compile(
            optimizer=self.optimizer,
            loss_fn=self.loss_fn
        )

        return model

    def train(self):
        history = self.model.fit(
            self.train_ds,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_data=self.test_ds,
            validation_steps=self.steps_per_epoch,
            callbacks=self.monitors
        )

        return history

    def evaluate(self, content: str, style: str, save_filename='img.jpg', size=(256, 256)):
        content_image = decode_and_resize(content)
        content_image = tf.convert_to_tensor(
            tf.reshape(
                content_image, (1, *content_image.shape)
            ), dtype=tf.float32
        )

        style_image = decode_and_resize(style)
        style_image = tf.convert_to_tensor(
            tf.reshape(
                style_image, (1, *style_image.shape)
            ), dtype=tf.float32
        )

        recon_image = self.model.inference(content_image, style_image)

        if self.mode == 'inference':
            keras.preprocessing.image.save_img(f'{self.model_name}/inferences/{save_filename}', recon_image[0])

        if self.mode == 'evaluate':
            save_name = content.rsplit(".", 1)[0].split('/')[-1] + '_stylized_' + save_filename
            keras.preprocessing.image.save_img(f'{self.model_name}/evaluates/{save_name}', recon_image[0])

        keras.preprocessing.image.save_img(f'{save_filename}', recon_image[0])

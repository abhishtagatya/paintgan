"""
    Gatys et al.
    Paper : https://arxiv.org/abs/1508.06576

    A Neural Algorithm for Artistic Style

    Tensorflow Implementation

    Original Implementation : -
    Reference : https://www.tensorflow.org/tutorials/generative/style_transfer
"""

import csv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt

from algorithm.base import Algorithm

from algorithm.gatys_comp.preprocessing import preprocess_image
from algorithm.gatys_comp.func import style_loss_fn, content_loss_fn, clip_0_1
from algorithm.gatys_comp.monitor import CSVLogger
from algorithm.gatys_comp.model import GatysFeatureExtractor


class Gatys(Algorithm):

    def __init__(self,
                 content_dir='',
                 style_dir='',
                 epochs=1,
                 batch_size=1,
                 steps_per_epoch=100,
                 image_size=(256, 256),
                 content_weight=1e4,
                 style_weight=1e-2,
                 checkpoint=None,
                 mode='train'
                 ):
        super(Gatys, self).__init__(content_dir, style_dir, epochs, batch_size, image_size, mode)
        self._create_result_folder()
        self.steps_per_epoch = steps_per_epoch

        self.optimizer = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

        self.content_weight = content_weight
        self.style_weight = style_weight

        self.content_layers = ['block5_conv2']

        self.style_layers = ['block1_conv1',
                             'block2_conv1',
                             'block3_conv1',
                             'block4_conv1',
                             'block5_conv1']

        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)

        self.content_image = preprocess_image(self.content_dir)
        self.style_image = preprocess_image(self.style_dir)

        self.csv_log = CSVLogger(self.model_name, self.style_dir, self.epochs, self.epochs)
        self.train_log = []

        self.build_model()

        # self.csv_monitor = CSVLogger(
        #     f'{self.model_name}-{self.epochs}-{self.steps_per_epoch}.csv',
        #     append=True,
        #     separator=';'
        # )

    def build_model(self):
        self.extractor = GatysFeatureExtractor(self.style_layers, self.content_layers)

        self.style_targets = self.extractor(self.style_image)['style']
        self.content_targets = self.extractor(self.content_image)['content']

        self.train_style_loss_metric = keras.metrics.Mean(name="style_loss")
        self.train_content_loss_metric = keras.metrics.Mean(name="content_loss")
        self.train_total_loss_metric = keras.metrics.Mean(name="total_loss")

    @tf.function()
    def train_step(
            self,
            image,
            total_variation_weight=30):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)

            style_outputs = outputs['style']
            content_outputs = outputs['content']

            style_loss = style_loss_fn(style_outputs, self.style_targets, self.num_style_layers)
            content_loss = content_loss_fn(content_outputs, self.content_targets, self.num_content_layers)

            total_loss = style_loss + content_loss
            total_loss += total_variation_weight * tf.image.total_variation(image)

        gradients = tape.gradient(total_loss, image)
        self.optimizer.apply_gradients([(gradients, image)])

        image.assign(clip_0_1(image))

        self.train_style_loss_metric.update_state(style_loss)
        self.train_content_loss_metric.update_state(content_loss)
        self.train_total_loss_metric.update_state(total_loss)

        return {
            "style_loss": self.train_style_loss_metric.result(),
            "content_loss": self.train_content_loss_metric.result(),
            "total_loss": self.train_total_loss_metric.result()
        }

    def train(self):

        image = tf.Variable(self.content_image)

        for epoch in range(self.epochs):

            if self.mode == 'train':
                print(f'\nEpoch {epoch + 1}/{self.epochs}')
                pb_i = Progbar(self.steps_per_epoch, stateful_metrics=[
                    'style_loss', 'content_loss', 'total_loss'
                ])

            for steps in range(self.steps_per_epoch):
                self.train_step(image)

                if self.mode == 'train':
                    metric_values = [
                        ('style_loss', self.train_style_loss_metric.result()),
                        ('content_loss', self.train_content_loss_metric.result()),
                        ('total_loss', self.train_total_loss_metric.result())
                    ]
                    pb_i.add(1, values=metric_values)

            if self.mode == "train":
                self.train_log.append([
                    epoch,
                    self.train_content_loss_metric.result(),
                    self.train_style_loss_metric.result(),
                    self.train_total_loss_metric.result()
                ])

            self.train_style_loss_metric.reset_states()
            self.train_content_loss_metric.reset_states()
            self.train_total_loss_metric.reset_states()

            # Train Monitors
            if self.mode == 'train':

                fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
                ax[0].imshow(tf.squeeze(self.style_image, axis=0))
                ax[0].set_title(f"Style: {epoch + 1:03d}")

                ax[1].imshow(tf.squeeze(self.content_image, axis=0))
                ax[1].set_title(f"Content: {epoch + 1:03d}")

                ax[2].imshow(tf.squeeze(image, axis=0))
                ax[2].set_title(f"{self.model_name}: {epoch + 1:03d}")

                plt.savefig(f'{self.model_name}/results/{self.model_name}_{epoch + 1}.png', format='png')
                plt.show()

        if self.mode == 'train':
            self.csv_log.compile(self.train_log)

        return image

    def evaluate(self, save_filename='img.jpg'):
        recon_image = self.train()
        recon_image = tf.squeeze(recon_image, axis=0)

        if self.mode == 'inference':
            keras.preprocessing.image.save_img(f'{self.model_name}/inferences/{save_filename}', recon_image)

        if self.mode == 'evaluate':
            save_name = self.content_dir.rsplit(".", 1)[0].split('/')[-1] + '_stylized_' + save_filename
            keras.preprocessing.image.save_img(f'{self.model_name}/evaluates/{save_name}', recon_image)

        keras.preprocessing.image.save_img(f'{save_filename}', recon_image)




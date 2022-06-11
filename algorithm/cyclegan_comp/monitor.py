import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt


class CheckpointMonitor(tf.keras.callbacks.Callback):

    def __init__(self, model_name, domain, checkpoint_per=10):
        self.model_name = model_name
        self.domain = domain
        self.checkpoint_per = checkpoint_per

    def on_epoch_end(self, epoch, logs=None):
        # Saving model checkpoint
        if (epoch + 1) % self.checkpoint_per == 0:
            self.checkpoint_manager.save()
            # self.model.save(f'{self.model_name}/model_checkpoints/{self.model_name}-{self.domain}_{epoch + 1}.ckpt')


class DisplayMonitor(tf.keras.callbacks.Callback):

    def __init__(self, model_name, domain, dataset, image_size=(256, 26, 3)):
        self.model_name = model_name
        self.domain = domain
        self.dataset = dataset
        self.image_size = image_size

        self.content, self.style = self.select_image = self.dataset.take(1).get_single_element()

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(1, 3, figsize=(12, 12))

        # Create the img
        pred = self.model.gen_G(self.content)[0].numpy()
        pred = (pred * 127.5 + 127.5).astype(np.uint8)
        img = (self.content[0] * 127.5 + 127.5).numpy().astype(np.uint8)
        img_b = (self.style[0] * 127.5 + 127.5).numpy().astype(np.uint8)

        # Plot the style, content, image
        ax[0].imshow(img_b)
        ax[0].set_title(f"Style: {epoch + 1:03d}")

        ax[1].imshow(img)
        ax[1].set_title(f"Content: {epoch + 1:03d}")

        ax[2].imshow(pred)
        ax[2].set_title(f"Translated: {epoch + 1:03d}")

        plt.savefig(f'{self.model_name}/model_results/{self.model_name}-{self.domain}_{epoch + 1}.png', format='png')

        plt.show()
        plt.close()

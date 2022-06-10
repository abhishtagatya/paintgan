import tensorflow as tf
import matplotlib.pyplot as plt

from algorithm.ada_in_comp.func import ada_in_func


class CheckpointMonitor(tf.keras.callbacks.Callback):

    def __init__(self, model_name, checkpoint_per=10):
        self.model_name = model_name
        self.checkpoint_per = checkpoint_per

    def on_epoch_end(self, epoch, logs=None):
        # Saving model checkpoint
        if (epoch + 1) % self.checkpoint_per == 0:
            self.model.save_weights(f'{self.model_name}/model_checkpoints/{self.model_name}_{epoch + 1}.ckpt')


class DisplayMonitor(tf.keras.callbacks.Callback):

    def __init__(self, model_name, dataset):
        self.model_name = model_name
        self.dataset = dataset

        self.test_content, self.test_style = next(iter(self.dataset))

    def on_epoch_end(self, epoch, logs=None):
        # Encode the style and content image

        test_style_encoded = self.model.encoder(self.test_style)
        test_content_encoded = self.model.encoder(self.test_content)

        test_t = ada_in_func(style=test_style_encoded, content=test_content_encoded)
        test_recon_image = self.model.decoder(test_t)

        # Plot the style, content, image
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        ax[0].imshow(tf.keras.preprocessing.image.array_to_img(self.test_style[0]))
        ax[0].set_title(f"Style: {epoch + 1:03d}")

        ax[1].imshow(tf.keras.preprocessing.image.array_to_img(self.test_content[0]))
        ax[1].set_title(f"Content: {epoch + 1:03d}")

        ax[2].imshow(tf.keras.preprocessing.image.array_to_img(test_recon_image[0]))
        ax[2].set_title(f"{self.model_name}: {epoch + 1:03d}")

        plt.savefig(f'{self.model_name}/model_results/{self.model_name}_{epoch + 1}.png', format='png')

        plt.show()
        plt.close()

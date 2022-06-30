import numpy as np

import matplotlib.pyplot as plt


class CheckpointMonitor:

    def __init__(self, model, model_name, domain, checkpoint_per=10):
        self.model = model
        self.model_name = model_name
        self.domain = domain
        self.checkpoint_per = checkpoint_per

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.checkpoint_per == 0:
            self.model.checkpoint_manager.save()


class DisplayMonitor:

    def __init__(self, model, model_name, domain, dataset, image_size=(256, 256, 3)):
        self.model = model
        self.model_name = model_name
        self.domain = domain
        self.dataset = dataset
        self.image_size = image_size

        self.content, self.style = self.dataset.take(1).get_single_element()

    def on_epoch_end(self, epoch, logs=None):

        fake_B = self.model.generator_AB.predict(self.content)
        fake_A = self.model.generator_BA.predict(self.style)

        fake_B = (fake_B[0] * 127.5 + 127.5).astype(np.uint8)
        fake_A = (fake_A[0] * 127.7 + 127.5).astype(np.uint8)
        img_A = np.array(self.content[0] * 127.5 + 127.5).astype(np.uint8)
        img_B = np.array(self.style[0] * 127.5 + 127.5).astype(np.uint8)

        _, ax = plt.subplots(1, 4, figsize=(12, 12))

        # Plot the style, content, image
        ax[0].imshow(img_B)
        ax[0].set_title(f"Style: {epoch + 1:03d}")

        ax[1].imshow(img_A)
        ax[1].set_title(f"Content: {epoch + 1:03d}")

        ax[2].imshow(fake_B)
        ax[2].set_title(f"Translated A > B: {epoch + 1:03d}")

        ax[3].imshow(fake_A)
        ax[3].set_title(f"Translated B < A: {epoch + 1:03d}")

        plt.savefig(f'{self.model_name}/results/{self.model_name}-{self.domain}_{epoch + 1}.png', format='png')

        plt.show()
        plt.close()
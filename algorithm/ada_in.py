"""
    AdaIN : Adaptive Instance Normalization
    Paper : https://arxiv.org/pdf/1703.06868v2.pdf

    Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization

    Tensorflow Implementation

    Original Implementation : https://github.com/xunhuang1995/AdaIN-style (Lua)
    Reference : https://keras.io/examples/generative/adain/#downloading-the-dataset-from-kaggle
"""
import tensorflow as tf
from keras.callbacks import CSVLogger

from algorithm.base import Algorithm

from algorithm.ada_in_comp.preprocessing import decode_and_resize
from algorithm.ada_in_comp.func import get_encoder, get_decoder, get_loss_net
from algorithm.ada_in_comp.monitor import DisplayMonitor, CheckpointMonitor
from algorithm.ada_in_comp.model import AdaptiveInstanceNorm
from util.data_loader import ArbitraryDataLoader


class AdaIN(Algorithm):

    def __init__(self,
                 content_dir,
                 style_dir,
                 epochs=1,
                 batch_size=1,
                 image_size=(256, 256),
                 style_weight=4.0,
                 checkpoint=None):
        super(AdaIN, self).__init__(content_dir, style_dir, epochs, batch_size, image_size)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.encoder = get_encoder(self.image_size)
        self.loss_net = get_loss_net(self.image_size)
        self.decoder = get_decoder()

        self.style_weight = style_weight

        self.data_loader = ArbitraryDataLoader(content_path=self.content_dir, style_path=self.style_dir)
        self.train_ds, self.test_ds = self.data_loader.as_dataset(
            preprocess_func=decode_and_resize,
            batch_size=self.batch_size
        )

        self.model = self.build_model()

        if checkpoint:
            self.model.load_weights(checkpoint)

        self.monitors = [
            # DisplayMonitor('model_results', self.test_ds),
            CheckpointMonitor('model_checkpoints', checkpoint_per=10),
            CSVLogger(f'{__name__}_p365-{self.epochs}-{self.batch_size}.csv', append=True, separator=';')
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
            batch_size=self.batch_size,
            validation_data=self.test_ds,
            callbacks=self.monitors
        )

        return history

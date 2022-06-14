import os
from tabulate import tabulate

import numpy as np
import pandas as pd

import tensorflow
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras import Model, Sequential
from tensorflow.keras.regularizers import l2

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DeceptionScore:

    def __init__(self,
                 model_name,
                 eval_dataset_file,
                 checkpoint='',
                 log=True):
        self.model_name = model_name
        self.eval_dataset = pd.read_hdf(eval_dataset_file, key='eval')

        if not checkpoint:
            raise ValueError('Checkpoint is required for this evaluation model')

        self.model = keras.models.load_model(checkpoint)

    @staticmethod
    def _preprocess_eval_set(eval_files):
        temp_list = []
        for file in eval_files:
            artist_label = file.split('_stylized_')[-1].replace('.jpg', '')
            temp_list.append((
                file, artist_label
            ))

        df = pd.DataFrame(temp_list, columns=['Path', 'Artist'])
        return df

    @staticmethod
    def _generate_dataset(dataframe, classes, x_col='Path', y_col='Artist'):
        gen_val = ImageDataGenerator(
            rescale=1. / 255
        )

        val_generator = gen_val.flow_from_dataframe(
            dataframe,
            x_col=x_col,
            y_col=y_col,
            target_size=(224, 224),
            color_mode='rgb',
            shuffle=False,
            batch_size=32,
            class_mode='categorical',
            classes=classes,
            validate_filenames=True,
            interpolation='nearest',
        )

        return val_generator

    def score(self, eval_dir, class_name='Artist'):
        eval_files = os.listdir(eval_dir)
        eval_dataset = self._preprocess_eval_set(eval_files)
        num_classes = len(eval_dataset[class_name].unique())

        eval_generator = self._generate_dataset(eval_dataset, num_classes)

        # Preprocess prediction
        y_pred = self.model.predict(eval_generator)
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        accuracy.update_state(eval_generator.classes, y_pred)

        # Logging
        content = [self.model_name, accuracy.result().numpy(), len(eval_files), num_classes]
        print(tabulate(content, headers=['Model', 'Score', 'Num. Files', 'Num. Class']))

    @staticmethod
    def build_eval_model(num_classes):
        vgg16c = tf.keras.Sequential()
        vgg16c.add(Input(shape=(224, 224, 3)))

        # First Block
        vgg16c.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
        vgg16c.add(BatchNormalization())
        vgg16c.add(Conv2D(filters=64, kernel_size=3, padding="same", activation="relu"))
        vgg16c.add(BatchNormalization())
        vgg16c.add(MaxPool2D(pool_size=2, strides=2, padding="same"))

        # Second Block
        vgg16c.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
        vgg16c.add(BatchNormalization())
        vgg16c.add(Conv2D(filters=128, kernel_size=3, padding="same", activation="relu"))
        vgg16c.add(BatchNormalization())
        vgg16c.add(MaxPool2D(pool_size=2, strides=2, padding="same"))

        # Third Block
        vgg16c.add(
            Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        vgg16c.add(BatchNormalization())
        vgg16c.add(
            Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        vgg16c.add(BatchNormalization())
        vgg16c.add(
            Conv2D(filters=256, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        vgg16c.add(BatchNormalization())
        vgg16c.add(MaxPool2D(pool_size=2, strides=2, padding="same"))

        # Fourth Block
        vgg16c.add(
            Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        vgg16c.add(BatchNormalization())
        vgg16c.add(
            Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        vgg16c.add(BatchNormalization())
        vgg16c.add(
            Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        vgg16c.add(BatchNormalization())
        vgg16c.add(MaxPool2D(pool_size=2, strides=2, padding="same"))

        # Fifth Block
        vgg16c.add(
            Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        vgg16c.add(BatchNormalization())
        vgg16c.add(
            Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        vgg16c.add(BatchNormalization())
        vgg16c.add(
            Conv2D(filters=512, kernel_size=3, padding="same", activation="relu", kernel_regularizer=l2(0.01)))
        vgg16c.add(BatchNormalization())
        vgg16c.add(MaxPool2D(pool_size=2, strides=2, padding="same"))

        # Fully Connected
        vgg16c.add(Flatten())
        vgg16c.add(Dense(units=4096, activation="relu"))
        vgg16c.add(Dropout(0.3))
        vgg16c.add(Dense(units=4096, activation="relu"))
        vgg16c.add(Dropout(0.3))
        vgg16c.add(Dense(units=num_classes, activation="softmax"))

        return vgg16c

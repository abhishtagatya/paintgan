import os
import glob
import random
from struct import unpack

import tensorflow as tf

"""
Mappings for JPEG Corruption Check

{
    0xffd8: "Start of Image",
    0xffe0: "Application Default Header",
    0xffdb: "Quantization Table",
    0xffc0: "Start of Frame",
    0xffc4: "Define Huffman Table",
    0xffda: "Start of Scan",
    0xffd9: "End of Image"
}
"""


class JPEG:

    def __init__(self, image_file):
        with open(image_file, 'rb') as f:
            self.img_data = f.read()

    def decode(self):
        data = self.img_data
        while True:
            marker, = unpack(">H", data[0:2])
            if marker == 0xffd8:
                data = data[2:]
            elif marker == 0xffd9:
                return
            elif marker == 0xffda:
                data = data[-2:]
            else:
                lenchunk, = unpack(">H", data[2:4])
                data = data[2 + lenchunk:]
            if len(data) == 0:
                break


class BaseDataLoader:
    """
    BaseDataLoader

    Base Class for any data loading techniques that is used by Algorithms.
    :param content_path: Path to Content Folder
    :param style_path: Path to Style Folder
    """

    def __init__(self, content_path: str, style_path: str):
        self.content_path = content_path
        self.style_path = style_path

    @staticmethod
    def remove_corrupt(path_list):
        copy_path_list = path_list

        for image in path_list:
            try:
                image_d = JPEG(image)
                image_d.decode()
            except:
                copy_path_list.remove(image)
        return copy_path_list

    def load_data(self, preprocessing_func=None, validation_split=0, auto_tune=True):
        return os.listdir(self.content_path), os.listdir(self.style_path)


class PairedDataLoader(BaseDataLoader):

    def __init__(self, content_path: str, style_path: str):
        super(PairedDataLoader, self).__init__(
            content_path,
            style_path,
        )


class ArbitraryDataLoader(BaseDataLoader):

    def __init__(self, content_path: str, style_path: str, recursive=True):
        super(ArbitraryDataLoader, self).__init__(
            content_path,
            style_path,
        )

        self.content_list = glob.glob(self.content_path + '/**/*.jpg', recursive=recursive)
        self.style_list = glob.glob(self.style_path + '/**/*.jpg', recursive=recursive)

        self.content_list = self.remove_corrupt(self.content_list)
        self.style_list = self.remove_corrupt(self.style_list)

        self.total_content = len(self.content_list)
        self.total_style = len(self.style_list)

    def as_dataset(self, preprocess_func, batch_size=1, val_split=0.2, auto_tune=tf.data.AUTOTUNE):
        """
        Load class as a Designated Dataset

        :param preprocess_func: Image Preprocessing Function
        :param batch_size: Batch Size of Dataset
        :param val_split: Split between Train and Validation
        :param auto_tune: Auto Tune Dataset
        :return: (Train, Validation) -> Tensorflow Dataset
        """

        # Splitting Content by Ratio
        content_train = self.content_list[:int(self.total_content * (1.0 - val_split))]
        content_val = self.content_list[int(self.total_content * (1.0 - val_split)):]
        style_train = self.style_list[:int(self.total_style * (1.0 - val_split))]
        style_val = self.style_list[int(self.total_style * (1.0 - val_split)):]

        train_content_ds = (
            tf.data.Dataset.from_tensor_slices(content_train)
                .map(preprocess_func, num_parallel_calls=auto_tune)
                .repeat()
        )

        val_content_ds = (
            tf.data.Dataset.from_tensor_slices(content_val)
                .map(preprocess_func, num_parallel_calls=auto_tune)
                .repeat()
        )

        train_style_ds = (
            tf.data.Dataset.from_tensor_slices(style_train)
                .map(preprocess_func, num_parallel_calls=auto_tune)
                .repeat()
        )

        val_style_ds = (
            tf.data.Dataset.from_tensor_slices(style_val)
                .map(preprocess_func, num_parallel_calls=auto_tune)
                .repeat()
        )

        train_ds = (
            tf.data.Dataset.zip((train_content_ds, train_style_ds))
                .shuffle(batch_size * 2)
                .batch(batch_size)
                .prefetch(auto_tune)
        )

        val_ds = (
            tf.data.Dataset.zip((val_content_ds, val_style_ds))
                .shuffle(batch_size * 2)
                .batch(batch_size)
                .prefetch(auto_tune)
        )

        return train_ds, val_ds


class DomainDataLoader(BaseDataLoader):

    def __init__(self, content_path, style_path, content_domain='', style_domain='', max_set=0, recursive=True):
        super(DomainDataLoader, self).__init__(
            content_path,
            style_path,
        )

        self.content_list = glob.glob(self.content_path + '/**/*.jpg', recursive=recursive)
        self.style_list = glob.glob(self.style_path + '/**/*.jpg', recursive=recursive)

        self.content_domain = content_domain
        self.style_domain = style_domain

        self.max_set = max_set

        if self.content_domain != '':
            self.content_list = self.domain_selection(self.content_list, self.content_domain)

        if self.style_domain != '':
            self.style_list = self.domain_selection(self.style_list, self.style_domain)

        if self.max_set > 0:
            self.content_list = random.choices(self.content_list, k=self.max_set)
            self.style_list = random.choices(self.style_list, k=self.max_set)

        self.content_list = self.remove_corrupt(self.content_list)
        self.style_list = self.remove_corrupt(self.style_list)

        self.total_content = len(self.content_list)
        self.total_style = len(self.style_list)

    @staticmethod
    def domain_selection(image_list, domain):
        new_list = []
        for image in image_list:
            if domain in image:
                new_list.append(image)
        return new_list

    def as_dataset(self,
                   preprocess_func_train,
                   preprocess_func_test,
                   batch_size=1,
                   buffer_size=1,
                   val_split=0.2,
                   auto_tune=tf.data.AUTOTUNE
                   ):
        """
            Load class as a Designated Dataset

            :param max_set: Maximum number of set chosen randomly
            :param preprocess_func_train: Image Preprocessing Function
            :param preprocess_func_test: Image Preprocessing Function
            :param batch_size: Batch Size of Dataset
            :param buffer_size: Buffer Size of Dataset
            :param val_split: Split between Train and Validation
            :param auto_tune: Auto Tune Dataset
            :return: (Train, Validation) -> Tensorflow Dataset
        """

        # Splitting Content by Ratio
        content_train = self.content_list[:int(self.total_content * (1.0 - val_split))]
        content_val = self.content_list[int(self.total_content * (1.0 - val_split)):]
        style_train = self.style_list[:int(self.total_style * (1.0 - val_split))]
        style_val = self.style_list[int(self.total_style * (1.0 - val_split)):]

        train_content_ds = (
            tf.data.Dataset.from_tensor_slices(content_train)
                .map(preprocess_func_train, num_parallel_calls=auto_tune)
                .cache()
                .shuffle(buffer_size)
                .batch(batch_size)
        )

        val_content_ds = (
            tf.data.Dataset.from_tensor_slices(content_val)
                .map(preprocess_func_test, num_parallel_calls=auto_tune)
                .cache()
                .shuffle(buffer_size)
                .batch(batch_size)
        )

        train_style_ds = (
            tf.data.Dataset.from_tensor_slices(style_train)
                .map(preprocess_func_train, num_parallel_calls=auto_tune)
                .cache()
                .shuffle(buffer_size)
                .batch(batch_size)
        )

        val_style_ds = (
            tf.data.Dataset.from_tensor_slices(style_val)
                .map(preprocess_func_test, num_parallel_calls=auto_tune)
                .cache()
                .shuffle(buffer_size)
                .batch(batch_size)
        )

        train_ds = (
            tf.data.Dataset.zip((train_content_ds, train_style_ds))
        )

        val_ds = (
            tf.data.Dataset.zip((val_content_ds, val_style_ds))
        )

        return train_ds, val_ds

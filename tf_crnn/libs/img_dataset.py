import random

import tensorflow as tf
import math
import os
import cv2
import numpy as np
import uuid
import sys

from tensorflow.python.framework import dtypes

sys.path.insert(0, './')
import libs.utils as utils

"""
    使用 Dataset api 并行读取图片数据
    参考：
        - 关于 TF Dataset api 的改进讨论：https://github.com/tensorflow/tensorflow/issues/7951
        - https://kratzert.github.io/2017/06/15/example-of-tensorflows-new-input-pipeline.html
        - https://stackoverflow.com/questions/47064693/tensorflow-data-api-prefetch
        - https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

    TL;DR
        Dataset.shuffle() 的 buffer_size 参数影响数据的随机性， TF 会先取 buffer_size 个数据放入 catch 中，再从里面选取
        batch_size 个数据，所以使用 shuffle 有两种方法：
            1. 每次调用 Dataset api 前手动 shuffle 一下 filepaths 和 labels
            2. Dataset.shuffle() 的 buffer_size 直接设为 len(filepaths)。这种做法要保证 shuffle() 函数在 map、batch 前调用

        Dataset.prefetch() 的 buffer_size 参数可以提高数据预加载性能，但是它比 tf.FIFOQueue 简单很多。
        tf.FIFOQueue supports multiple concurrent producers and consumers
"""


# noinspection PyMethodMayBeStatic
class ImgDataset:
    """
    Use tensorflow Dataset api to load images in parallel
    """

    def __init__(self, img_dir,
                 converter,
                 batch_size,
                 num_parallel_calls=6,
                 da=None,
                 shuffle=True,
                 data_ordered=False,
                 filter_labels=True):
        """
        :param img_dir:
        :param converter:
        :param batch_size:
        :param num_parallel_calls:
        :param shuffle:
        :param data_ordered:
        :param filter_labels: 把 label 中不包含在 chars 中的字符过滤掉
        """
        self.converter = converter
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.img_dir = img_dir
        self.img_dirname = os.path.basename(img_dir)
        self.shuffle = shuffle
        self.step = 0
        self.da = da

        if data_ordered:
            img_paths, labels = utils.load_ordered_labels_and_paths(img_dir)
        else:
            img_paths, labels = utils.load_labels_and_paths(img_dir)

        labels = utils.str_Q2B(labels, self.converter.chars)

        if filter_labels:
            labels = converter.filter_labels(labels)

        labels = utils.strip_space(labels)

        self.size = len(labels)

        dataset = self._create_dataset(img_paths, labels)
        iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                                   dataset.output_shapes)
        self.next_batch = iterator.get_next()
        self.init_op = iterator.make_initializer(dataset)

        self.num_batches = math.ceil(self.size / self.batch_size)

    def get_next_batch(self, sess, step=0):
        """return images and labels of a batch"""
        self.step = step
        img_batch, widths, labels, img_paths = sess.run(self.next_batch)
        labels = [x[0].decode() for x in labels]
        img_paths = [x[0].decode() for x in img_paths]

        img_paths, labels = self.filter_empty_labels(img_paths, labels)

        encoded_label_batch = self.converter.encode_list(labels)
        sparse_label_batch = self._sparse_tuple_from_label(encoded_label_batch)
        return img_batch, widths, sparse_label_batch, labels, img_paths

    def filter_empty_labels(self, img_paths, labels):
        tmp_labels = []
        tmp_img_paths = []
        for i, l in enumerate(labels):
            # label = l.replace(' ', '')
            if l != '':
                tmp_img_paths.append(img_paths[i])
                tmp_labels.append(l)
        return tmp_img_paths, tmp_labels

    def _sparse_tuple_from_label(self, sequences, default_val=-1, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
                      encode label, e.g: [2,44,11,55]
            default_val: value should be ignored in sequences
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            seq_filtered = list(filter(lambda x: x != default_val, seq))
            indices.extend(zip([n] * len(seq_filtered), range(len(seq_filtered))))
            values.extend(seq_filtered)

        indices = np.asarray(indices, dtype=np.int64)
        values = np.asarray(values, dtype=dtype)

        if len(indices) == 0:
            shape = np.asarray([len(sequences), 0], dtype=np.int64)
        else:
            shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

        return indices, values, shape

    def _create_dataset(self, img_paths, labels):
        img_paths = tf.convert_to_tensor(img_paths, dtype=dtypes.string)
        labels = tf.convert_to_tensor(labels, dtype=dtypes.string)

        d = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        if self.shuffle:
            d = d.shuffle(buffer_size=self.size)

        d = d.map(lambda img_path, label: tf.py_func(self._input_parser, [img_path, label],
                                                     [tf.float32, tf.int64, tf.string, tf.string]),
                  num_parallel_calls=self.num_parallel_calls)

        # tensorflow do padding on normalized image
        # so if we want padding black(zero), we should normalize here too
        d = d.padded_batch(self.batch_size,
                           padded_shapes=([32, None, 1], [None], [None], [None]),
                           padding_values=((255.0 - 128.0) / 128.0, np.int64(0), '', ''))
        # d = d.batch(self.batch_size)

        # d = d.repeat(self.num_epochs)
        d = d.prefetch(buffer_size=self.batch_size * 2)
        return d

    def _input_parser(self, img_path, label):
        img_path = img_path.decode()
        #print('img_path is :', img_path )
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if self.da is not None:
            img = self.da.run(img)

        img = img[:, :, np.newaxis]

        if img.shape[0] != 32:
            scale = 32. / img.shape[0]
            scaled_width = int(img.shape[1] * scale)
            img = cv2.resize(img, (scaled_width, 32), interpolation=cv2.INTER_LINEAR)
            img = img[:, :, np.newaxis]

        img = img.astype(np.float32)
        img = (img - 128.0) / 128.0

        width = img.shape[1]
        return img, [width], [label], [img_path]


if __name__ == '__main__':
    from libs.label_converter import LabelConverter

    demo_path = '/tf_crnn/data/demo2'
    chars_file = '/ocr_chars/chn.txt'
    epochs = 5
    batch_size = 2

    converter = LabelConverter(chars_file=chars_file)
    ds = ImgDataset(demo_path, converter, batch_size=batch_size)

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        for epoch in range(epochs):
            sess.run(ds.init_op)
            print('------------Epoch(%d)------------' % epoch)
            for batch in range(ds.num_batches):
                _, _, _, labels, _ = ds.get_next_batch(sess)
                print(labels[0])

from typing import List

import tensorflow as tf
import math
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, './')
import libs.utils as utils

TFRECORD_DATASET_FILE_NAME = "TFRecord_dataset"
DATA_NUM_FILE_NAME = 'dataset_num'


class TFRecordDataset:
    def __init__(self, train_dir,
                 converter,
                 batch_size,
                 shuffle_batch_size=1,
                 num_parallel_calls=6,
                 da=None,
                 filter_labels=True):
        self.filter_labels = filter_labels
        self.da = da
        self.num_parallel_calls = num_parallel_calls
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.converter = converter
        self.da = da

        dataset_file_paths = self.get_TFRecord_filepath(train_dir)
        self.size = self.get_data_size(train_dir)

        self.num_batches = math.ceil(self.size / self.batch_size)

        self.dataset = tf.data.TFRecordDataset(dataset_file_paths)
        self.dataset = self.dataset.shuffle(shuffle_batch_size)
        self.dataset = self.dataset.map(self._input_parser, num_parallel_calls=self.num_parallel_calls)
        self.dataset = self.dataset.padded_batch(self.batch_size,
                                                 padded_shapes=(
                                                     [None], [None], [None]),
                                                 padding_values=(
                                                     np.uint8(32), np.uint8(0), np.int32(0)))  # 32 空格的unicode编码
        self.dataset = self.dataset.prefetch(buffer_size=self.batch_size * 2)
        iterator = self.dataset.make_initializable_iterator()
        self.next_batch = iterator.get_next()
        self.init_op = iterator.make_initializer(self.dataset)

    def get_next_batch(self, sess):
        label_batch, image_raw_batch, image_raw_length_batch = sess.run(self.next_batch)

        label_batch = [x.tostring().decode() for x in label_batch]
        label_batch = utils.str_Q2B(label_batch, self.converter.chars)
        if self.filter_labels:
            label_batch = self.converter.filter_labels(label_batch)
        label_batch = utils.strip_space(label_batch)

        encoded_label_batch = self.converter.encode_list(label_batch)
        sparse_label_batch = self._sparse_tuple_from_label(encoded_label_batch)

        np_img_batch, width_batch = self._decode_raw_imgs(image_raw_batch, image_raw_length_batch)

        tmp_img_batch = []
        tmp_width_batch = []
        for np_img in np_img_batch:
            if self.da is not None:
                np_img = self.da.run(np_img)
            np_img = self.image_pre_process(np_img)
            tmp_img_batch.append(np_img)
            tmp_width_batch.append(np_img.shape[1])

        np_img_batch = tmp_img_batch
        width_batch = tmp_width_batch

        max_width = max(width_batch)

        image_padded_batch = []
        for np_img in np_img_batch:
            image_padded = self.pad_image(np_img, max_width, (255.0 - 128.0) / 128.0)
            image_padded_batch.append(image_padded)

        # 生成小文件时，文件名就是label,方便人工查看
        return image_padded_batch, width_batch, sparse_label_batch, label_batch, label_batch

    def _decode_raw_imgs(self, image_raw_batch, image_raw_length_batch) -> (List[np.ndarray], List[int]):
        """
        把 TFRecord 的 raw 数据转成 numpy 图像
        """
        np_imgs = []
        widths = []
        for image_raw, image_raw_length in zip(image_raw_batch, image_raw_length_batch):
            np_img = cv2.imdecode(np.fromstring(image_raw.tostring()[0:image_raw_length[0]], dtype=np.uint8),
                                  cv2.IMREAD_GRAYSCALE)

            np_imgs.append(np_img)
            widths.append(np_img.shape[1])
        return np_imgs, widths

    def image_pre_process(self, img_padded):
        image = img_padded[:, :, np.newaxis]
        if image.shape[0] != 32:
            scale = 32. / image.shape[0]
            scaled_width = int(image.shape[1] * scale)
            image = cv2.resize(image, (scaled_width, 32), interpolation=cv2.INTER_LINEAR)
            image = image[:, :, np.newaxis]
        image = image.astype(np.float32)
        image = (image - 128.0) / 128.0

        return image

    def pad_image(self, image, max_width, pad_value):
        res = np.tile(pad_value, (32, max_width, 1))
        w = image.shape[1]
        res[:, 0:w, :] = image
        return res

    def _input_parser(self, serialized_tfrecord):
        features = {
            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'image_raw_length': tf.FixedLenFeature([], tf.int64),
        }

        feature = tf.parse_single_example(serialized_tfrecord, features)

        image_raw = tf.decode_raw(feature['image_raw'], tf.uint8)

        label = tf.decode_raw(feature['label'], tf.uint8)

        image_raw_length = tf.cast(feature['image_raw_length'], tf.int32)

        return label, image_raw, [image_raw_length]

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

    def get_data_size(self, dir_name):
        p = Path(dir_name)
        dataset_nums_paths = list(p.glob('**/%s' % DATA_NUM_FILE_NAME))
        total_size = sum([int(it.read_text()) for it in dataset_nums_paths])
        return total_size

    def get_TFRecord_filepath(self, dir_name):
        p = Path(dir_name)
        dataset_paths = list(p.glob('**/%s' % TFRECORD_DATASET_FILE_NAME))
        dataset_paths = [str(it) for it in dataset_paths]
        return dataset_paths


if __name__ == '__main__':
    import sys

    sys.path.insert(0, './')
    from libs.label_converter import LabelConverter
    from libs.data_augumenter import DataAugumenter
    from libs.config import load_config

    converter = LabelConverter(chars_file='/Users/cwq/code/tf_crnn/data/ocr_chars/num_and_big_eng.txt')

    cfg = load_config('raw8_one_lstm')

    da = DataAugumenter(cfg)
    da = None

    ds = TFRecordDataset(train_dir='/Users/cwq/data/TFRecord/random0_val',
                         converter=converter,
                         batch_size=20, da=da,
                         num_parallel_calls=1)
    with tf.Session() as sess:
        ds.init_op.run()
        image_padded_batch, width_batch, sparse_label_batch, label_batch, _ = ds.get_next_batch(sess)

        for i, img in enumerate(image_padded_batch):
            print(label_batch[i])
            print(np.count_nonzero(img != 0))
            img = img * 128 + 128
            cv2.imshow('debug', img.astype(np.uint8))
            cv2.waitKey()

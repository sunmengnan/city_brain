# -*- coding: utf-8 -*-
import os
import random
import sys
from typing import List

import numpy as np
import tensorflow as tf
import cv2
from functools import reduce


def load_chars(filepath):
    if not os.path.exists(filepath):
        print("Chars file not exists. %s" % filepath)
        exit(1)

    ret = ''
    with open(filepath, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            ret += line[0]
    return ret


# https://stackoverflow.com/questions/49063938/padding-labels-for-tensorflow-ctc-loss
def dense_to_sparse(dense_tensor, sparse_val=-1):
    """Inverse of tf.sparse_to_dense.

    Parameters:
        dense_tensor: The dense tensor. Duh.
        sparse_val: The value to "ignore": Occurrences of this value in the
                    dense tensor will not be represented in the sparse tensor.
                    NOTE: When/if later restoring this to a dense tensor, you
                    will probably want to choose this as the default value.
    Returns:
        SparseTensor equivalent to the dense input.
    """
    with tf.name_scope("dense_to_sparse"):
        sparse_inds = tf.where(tf.not_equal(dense_tensor, sparse_val),
                               name="sparse_inds")
        sparse_vals = tf.gather_nd(dense_tensor, sparse_inds,
                                   name="sparse_vals")
        dense_shape = tf.shape(dense_tensor, name="dense_shape",
                               out_type=tf.int64)
        return tf.SparseTensor(sparse_inds, sparse_vals, dense_shape)


def check_dir_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def restore_ckpt(sess, saver, checkpoint_dir):
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    try:
        saver.restore(sess, ckpt)
        print('Restore checkpoint from {}'.format(ckpt))
    except Exception as e:
        print(e)
        print("Can not restore from {}".format(checkpoint_dir))
        exit(-1)


def count_tf_params():
    """print number of trainable variables"""

    def size(v):
        return reduce(lambda x, y: x * y, v.get_shape().as_list())

    n = sum(size(v) for v in tf.trainable_variables())
    print("Tensorflow Model size: %dK" % (n / 1000,))
    return n


def get_img_paths_and_labels2(img_dir):
    """label 位于同名 txt 文件中"""
    img_paths = []
    labels = []

    def read_label(p):
        with open(p, mode='r', encoding='utf-8') as f:
            data = f.read()
        return data

    for root, sub_folder, file_list in os.walk(img_dir):
        for idx, file_name in enumerate(sorted(file_list)):
            if file_name.endswith('.jpg') and os.path.exists(os.path.join(img_dir, file_name)):
                image_path = os.path.join(root, file_name)
                img_paths.append(image_path)
                label_path = os.path.join(root, file_name[:-4] + '.txt')
                labels.append(read_label(label_path))
            else:
                print('file not found: {}'.format(file_name))

    return img_paths, labels


def get_img_paths_and_label_paths(img_dir, img_count):
    img_paths = []
    label_paths = []
    for i in range(img_count):
        base_name = "{:08d}".format(i)
        img_path = os.path.join(img_dir, base_name + ".jpg")
        label_path = os.path.join(img_dir, base_name + ".txt")
        img_paths.append(img_path)
        label_paths.append(label_path)

    return img_paths, label_paths


def load_labels(filepath, img_num=None):
    """
    Load labels from txt file line by line
    """
    if not os.path.exists(filepath):
        print("Label file not exists. %s" % filepath)
        exit(1)

    with open(filepath, 'r', encoding='utf-8') as f:
        labels = f.readlines()

    if img_num and img_num <= len(labels):
        labels = labels[0:img_num]

    # 移除换行符、首尾空格
    labels = [l[:-1].strip() for l in labels]
    return labels


def build_img_paths(img_dir, img_count):
    """
    Image name should be eight length with continue num. e.g. 00000000.jpg, 00000001.jpg
    """
    img_paths = []
    for i in range(img_count):
        base_name = "{:08d}".format(i)
        img_path = os.path.join(img_dir, base_name + ".jpg")
        img_paths.append(img_path)

    return img_paths


def list_valid_sub_name(dirname):
    """
    返回一个目录下的文件、文件夹名称，会过滤掉以 . 开头的目录，比如 .DS_Store
    """
    # docker image's LANG is ascii, so we need to encode dirname and decode it to utf-8
    names = os.listdir(str.encode(dirname))
    names = [n.decode('utf-8') for n in names]
    names = list(filter(lambda x: not x.startswith('.'), names))
    return names


def load_ordered_labels_and_paths(img_dir):
    """
    图片的文件名应该是 8 位连续数字：00000000.jpg, 00000001.jpg
    对应的 label 存在同一目录下的 labels.txt 里，每一行按顺序与图片对应
    支持二级目录
    """
    if os.path.exists(os.path.join(img_dir, 'labels.txt')):
        labels = load_labels(os.path.join(img_dir, 'labels.txt'))
        img_paths = build_img_paths(img_dir, len(labels))
        return img_paths, labels

    sub_names = list_valid_sub_name(img_dir)

    img_paths = []
    labels = []
    for sub_name in sub_names:
        sub_name_with_path = os.path.join(img_dir, sub_name)
        if is_dir(sub_name_with_path):
            sub_labels = load_labels(os.path.join(sub_name_with_path, 'labels.txt'))
            sub_img_paths = build_img_paths(sub_name_with_path, len(sub_labels))
            img_paths.extend(sub_img_paths)
            labels.extend(sub_labels)

    return img_paths, labels


def load_labels_and_paths(img_dir):
    """
    label 位于文件名中，以三个下划线分隔
    支持二级目录
    """
    img_paths = []
    labels = []

    sub_names = list_valid_sub_name(img_dir)

    for sub_name in sub_names:
        sub_name_with_path = os.path.join(img_dir, sub_name)

        if is_dir(sub_name_with_path):
            filenames = list_valid_sub_name(sub_name_with_path)
            for filename in filenames:
                image_path = os.path.join(sub_name_with_path, filename)
                img_paths.append(image_path)

                labels.append(parse_filename(filename))
        else:
            filename = sub_name_with_path
            img_paths.append(filename)
            labels.append(parse_filename(filename))

    return img_paths, labels


def parse_filename(filename):
    # test___imgname.png
    label = filename[:-4].split('/')[-1].split('___')[0]

    # tianrang use _ as space
    label = label.replace('_', ' ')
    return label


def is_dir(pathname):
    # we should use os.path.isdir, but in out docker, LOCAL LANG is 'ascii'
    # it will cause os.path.isdir failed
    if '.' in pathname.split('/')[-1]:
        return False
    else:
        return True


def remove_all_symbols(txt):
    """
    移除字符串中所有的符号
    :param txt:
    :return:
    """
    symbols = ",.<>/?;:'\"[]{}!@#$%^&*()-=+\|，。；：‘”？》《！（）"
    res_txt = ""
    for c in txt:
        if c not in symbols:
            res_txt += c
    return res_txt


def str_Q2B(labels: List[str], charset=''):
    """
    https://www.jianshu.com/p/a5d96457c4a4
    把不包含在字符集中的全角符号转换成半角符号
    :param labels: List[str]
    :param charset: 字符集
    :return:
    """
    out_labels = []
    for label in labels:
        res = label_Q2B(charset, label)
        out_labels.append(res)
    return out_labels


def strip_space(labels):
    out = []
    for label in labels:
        out.append(label.strip(' '))
    return out


def label_Q2B(charset, label):
    res = ''
    for char in label:
        if char not in charset:
            inside_code = ord(char)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            res += chr(inside_code)
        else:
            res += char
    return res


def prob(percent):
    """
    percent: 0 ~ 1, e.g: 如果 percent=0.1，有 10% 的可能性
    """
    assert 0 <= percent <= 1
    if random.uniform(0, 1) <= percent:
        return True
    return False


if __name__ == '__main__':
    print(str_Q2B(['：（），！？；／【】「」《》'])[0] == ':(),!?;/【】「」《》')

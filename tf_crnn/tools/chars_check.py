"""
Tools to check whether one chars file support all chars in test dataset
"""
import argparse
from collections import defaultdict

from libs import utils
from libs.utils import load_labels, load_labels_and_paths
import os


def main(args):
    chars = load_labels(args.chars_file)
    _, labels = load_labels_and_paths(args.data_dir)

    labels = utils.str_Q2B(labels)

    label_str = ''.join(labels)
    label_chars = set(label_str)

    print("Chars in chars_file: %d" % len(chars))
    print("Chars in test dataset: %d" % len(label_chars))

    print("Chars not in chars_file:")
    for c in label_chars:
        if c not in chars:
            print(c)
    print('\n')

    sub_data_dirs = []
    sub_names = utils.list_valid_sub_name(args.data_dir)
    for sub_name in sub_names:
        sub_name_with_path = os.path.join(args.data_dir, sub_name)
        if utils.is_dir(sub_name_with_path):
            sub_data_dirs.append(sub_name_with_path)

    error_chars = defaultdict(int)
    print("Labels contains not supported chars by chars_file:")
    for sub_data_dir in sub_data_dirs:
        _, sub_labels = load_labels_and_paths(sub_data_dir)

        # sub_labels = utils.str_Q2B(sub_labels)

        for label in sub_labels:
            for c in label:
                if c not in chars:
                    error_chars[c] += 1

        print("{}:".format(sub_data_dir))
        for c, num in error_chars.items():
            print("{} {}".format(c, num))
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--chars_file', default='./data/ocr_chars/chn.txt')
    parser.add_argument('--data_dir', default='/Users/cwq/data/crnn_test_data', help='Dataset in sub dir structure')

    args, _ = parser.parse_known_args()
    main(args)

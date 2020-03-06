"""
读取数据集的 labels 信息，统计字符的评率，可以用来比较不同训练集之间的字符频率分布，或者是训练集和测试集之间的字符分布情况
"""
import os
import argparse
import sys
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, './')
from libs import utils


def analyze_labels(path: str):
    char_count_dict = defaultdict(int)

    with open(path, mode='r', encoding='utf-8') as f:
        data = f.readlines()
        # 移除换行符、首尾空格
        data = map(lambda l: l[:-1].strip(), data)
        data = ''.join(data)
        total_chars_count = len(data)

        for c in data:
            char_count_dict[c] += 1

    return char_count_dict, total_chars_count


def print_info(chars_count_list, total_chars_count, name, output_dir, max_count=15):
    file_str = ''

    name_str = "Info for %s" % name
    print(name_str)
    file_str += (name_str + '\n')

    total_str = "Total chars count %d" % total_chars_count
    print(total_str)
    file_str += (total_str + '\n')

    freqs = list(map(lambda x: x[1] / total_chars_count, chars_count_list))
    avg_freq = np.mean(freqs)
    std = np.std(freqs)
    avg_freq_str = "Average frequence: %f +- %f %%" % (avg_freq, std)
    print(avg_freq_str)
    file_str += (avg_freq_str + '\n')

    above_avg_freq = sum(map(lambda x: x > avg_freq, freqs))
    above_avg_freq_str = "Chars freq in training label above average: %d" % above_avg_freq
    print(above_avg_freq_str)
    file_str += (above_avg_freq_str + '\n')

    # save result to txt:
    with open(os.path.join(output_dir, '{}.txt'.format(name)), mode='w') as f:
        f.write(file_str)
        for index, (k, v) in enumerate(chars_count_list):
            f.write("%s %f%% %d\n" % (k, v / total_chars_count, chars_count_list[index][1]))

    print("Top %d" % max_count)
    count = 0
    for index, (k, v) in enumerate(chars_count_list):
        print("%s %f%% %d" % (k, v / total_chars_count, chars_count_list[index][1]))
        count += 1
        if count > max_count:
            break

    print("Bottom %d" % max_count)
    count = 0
    reversed_list = list(reversed(chars_count_list))
    for index, (k, v) in enumerate(reversed_list):
        print("%s %f%% %d" % (k, v / total_chars_count, reversed_list[index][1]))
        count += 1
        if count > max_count:
            break

    return avg_freq, above_avg_freq


def show_plot(log=False):
    if log:
        plt.yscale('log', nonposy='clip')
    plt.ylabel('Count')
    plt.legend()
    plt.show()


def process_dir(label_dir, log=False):
    label_paths = glob.glob(label_dir + '/*.txt')

    for p in label_paths:
        name = p.split('/')[-1].split('.')[0]
        chars_count_list, total_chars_count = analyze_labels(p)
        print_info(chars_count_list, total_chars_count, name)

        y = list(map(lambda x: x[1], chars_count_list))
        plt.plot(y, label=name)

    show_plot(log)


def process_file(label_file: str, log=False):
    name = "label"
    chars_count_dict, total_chars_count = analyze_labels(label_file)

    # 降序
    chars_count_list = list(sorted(chars_count_dict.items(), key=lambda x: x[1], reverse=True))

    print_info(chars_count_list, total_chars_count, name)

    y = list(map(lambda x: x[1], chars_count_list))

    plt.plot(y, label=name)
    show_plot(log)


def main(args):
    if args.data_ordered:
        img_paths, labels = utils.load_ordered_labels_and_paths(args.data_dir)
    else:
        img_paths, labels = utils.load_labels_and_paths(args.data_dir)

    chars_count_dict = defaultdict(int)
    labels_str = ''.join(labels)

    for c in labels_str:
        chars_count_dict[c] += 1

    # 降序
    chars_count_list = list(sorted(chars_count_dict.items(), key=lambda x: x[1], reverse=True))

    print_info(chars_count_list, len(labels_str), args.tag, args.output_dir)

    y = list(map(lambda x: x[1], chars_count_list))

    if args.show:
        plt.plot(y, label=args.tag)
        show_plot(args.log_scale)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/Users/cwq/data/crnn_test_data/0_all')
    parser.add_argument('--output_dir', type=str, default='/Users/cwq/data/char_freqs')
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--data_ordered', action='store_true', default=False)
    parser.add_argument('--load_sub_dir', action='store_true', default=False)
    parser.add_argument('--log_scale', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)

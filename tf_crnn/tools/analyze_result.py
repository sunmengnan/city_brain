"""
比较训练用label
读取两个 labels 文件，比较每一行的字符串真实值和识别值，
统计真实值与识别值长度相同的时结果中出错的字符串的数量
"""
import argparse
import os
import sys
from collections import defaultdict
from os.path import join as opj
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('./')
from libs.utils import load_chars
from libs import utils


class CharResult(object):
    def __init__(self, charsets):
        self.charsets = charsets
        self.total = {}  # 保存所有字符的 count
        self.count = {}  # 保存被识别错的字
        self.wrong_count = {}  # 保存每一个被识别错的字识别成了哪个字

        self.count = self.init_char_map()
        self.total = self.init_char_map()

        for c in self.charsets:
            self.wrong_count[c] = self.init_char_map()

    def init_char_map(self):
        out = {}
        for c in self.charsets:
            out[c] = 0
        return out

    def add(self, c):
        self.count[c] += 1

    def add_total(self, c):
        self.total[c] += 1

    def add_wrong(self, c, wrong_c):
        self.wrong_count[c][wrong_c] += 1

    def get_ordered_chars(self):
        out = []
        for k, v in sorted(self.count.items(), key=itemgetter(1), reverse=True):
            out.append(k)
        return out

    def _get_wrong_chars(self, c, top=3):
        chars = self.wrong_count[c]
        out = ""
        count = 0
        for k, v in sorted(chars.items(), key=itemgetter(1), reverse=True):
            if v != 0:
                out += "{} {} ".format(k, v)
            count += 1
            if count == top:
                break
        return out

    def get_char_str(self, c):
        count = self.count[c]
        if self.total[c] == 0:
            error_rate = 0
        else:
            error_rate = count / self.total[c] * 100
        return "%d/%d %.2f%%; %s\n" % (count, self.total[c], error_rate, self._get_wrong_chars(c))


def save_length_not_match(file_name, t_not_match, p_not_match):
    with open(file_name, mode='w', encoding='utf-8') as f:
        for i in range(len(t_not_match)):
            f.write("%s %s\n" % (t_not_match[i], p_not_match[i]))


def save_result(filepath, chars_count_dict, total_chars_count, char_result):
    freqs = list(map(lambda x: x / total_chars_count * 100, chars_count_dict.values()))
    avg_freq = np.mean(freqs)
    above_avg_freq = sum(map(lambda x: x > avg_freq, freqs))

    sorted_chars_count_list = sorted_by_chars(chars_count_dict, char_result.get_ordered_chars())
    with open(filepath, mode='w', encoding='utf-8') as f:
        f.write("avg freq: %f%%\n" % avg_freq)
        f.write("above avg freq chars: %d\n" % above_avg_freq)
        f.write("char  train_freq  error  error_rate  error_chars\n")
        for c, count in sorted_chars_count_list:
            f.write("%s   %f%%   %s" % (c, count / total_chars_count * 100, char_result.get_char_str(c)))


def compare(targets, predicts, charsets):
    t_result = CharResult(charsets)
    p_result = CharResult(charsets)

    t_not_match = []
    p_not_match = []
    for i in range(len(targets)):
        t_label = targets[i]
        p_label = predicts[i]
        if len(t_label) != len(p_label):
            t_not_match.append(t_label)
            p_not_match.append(p_label)
            continue

        for k in range(len(t_label)):
            t_result.add_total(t_label[k])
            p_result.add_total(p_label[k])
            if t_label[k] != p_label[k]:
                t_result.add(t_label[k])
                t_result.add_wrong(t_label[k], p_label[k])

                p_result.add(p_label[k])
                p_result.add_wrong(p_label[k], t_label[k])
    return t_result, p_result, t_not_match, p_not_match


def sorted_by_chars(chardict, chars):
    out = []
    for c in chars:
        if c not in chardict.keys():
            out.append((c, 0))
        else:
            out.append((c, chardict[c]))
    return out


def show_plot(imgpath, chars_count_dict, t_result):
    x = []
    y = []
    # 升序
    chars_count_list = list(sorted(chars_count_dict.items(), key=lambda a: a[1]))
    for c, v in chars_count_list:
        if c in t_result.count.keys() and (t_result.total[c] != 0):
            x.append(v)
            y.append(1 - (t_result.count[c] / t_result.total[c]))

    fig = plt.figure()
    plt.xscale('log')
    plt.xlabel('Char count in training label (log scale)')
    plt.ylabel('Char correct rate')

    # result.jpg：x 坐标（字符在 train_labels.txt 中出现的频率），y 坐标（预测结果中字符的正确率 1-error_rate）
    plt.scatter(x, y, s=5)
    fig.savefig(imgpath, )
    plt.show()


def load_gt_pred(filepath):
    targets = []
    predicts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        labels = f.readlines()
        labels = [l.strip() for l in labels]
        for label in labels:
            tmp = label.split('__$__')

            target = tmp[0]
            predict = tmp[1]

            targets.append(target)
            predicts.append(predict)
    return targets, predicts


def main(args):
    targets, predicts = load_gt_pred(args.gt_pred)

    assert len(targets) == len(predicts)

    print("Compare gt and predict...")
    charsets = load_chars(args.chars_file)
    t_result, p_result, t_not_match, p_not_match = compare(targets, predicts, charsets)

    print("Analyze training labels...")
    _, labels = utils.load_ordered_labels_and_paths(args.train_dir)
    labels = utils.str_Q2B(labels, charsets)

    chars_count_dict = defaultdict(int)
    labels_str = ''.join(labels)
    for c in labels_str:
        chars_count_dict[c] += 1
    total_chars_count = len(labels_str)

    print("Save result...")
    save_result(args.out_char_freq_check, chars_count_dict, total_chars_count, t_result)

    save_length_not_match(args.out_length_not_match, t_not_match, p_not_match)

    print("Show plot...")
    show_plot(args.out_img_path, chars_count_dict, t_result)


if __name__ == '__main__':
    """
    输入：
    - gt_and_pred.txt：跑 infer.py 会保存该文件，每一行的格式为 {gt}__$__{pred}
    
    输出：
    - char_frequency_check.txt: 最终结果
    - target_error_chars.txt: 真值中被识别错误的字符
    - predict_error_chars.txt:
    - result.jpg：x 坐标（字符在 train_labels.txt 中出现的频率），y 坐标（预测结果中字符的正确率 1-error_rate）
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_pred', type=str, default='')
    parser.add_argument('--train_dir', type=str, default='/disk2/cwq/crnn_train_data', help='训练集目录，用来统计字符在训练集中出现的频率')
    parser.add_argument('--output_dir', type=str, default='./output/analyze')
    parser.add_argument('--tag', type=str, default='default')
    parser.add_argument('--chars_file', type=str, default='./data/ocr_chars/chn.txt')

    args, _ = parser.parse_known_args()

    args.save_dir = opj(args.output_dir, args.tag)
    args.out_char_freq_check = opj(args.save_dir, 'char_freq_check.txt')
    args.out_length_not_match = opj(args.save_dir, 'length_not_match.txt')
    args.out_img_path = opj(args.save_dir, 'result.png')

    if not os.path.exists(args.gt_pred):
        print("gt_pred file  not exist")
        exit(-1)

    if not os.path.exists(args.train_dir):
        print("Train dir not exist")
        exit(-1)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    main(args)

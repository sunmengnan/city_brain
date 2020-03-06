"""
把 label 处于文件名中数据集转换到有序格式的数据集
"""
import os
import cv2
import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, './')
from libs import utils


def main(args):
    img_paths, labels = utils.load_labels_and_paths(args.input_dir)
    print("Total img: %d" % len(img_paths))

    out_labels = []

    for i in tqdm(range(len(img_paths))):
        save_img_filename = os.path.join(args.output_dir, "{:08d}.jpg".format(i))
        img = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
        if img is None or labels[i] == '':
            continue

        if img.shape[0] != 32:
            scale = 32. / img.shape[0]
            scaled_width = int(img.shape[1] * scale)
            img = cv2.resize(img, (scaled_width, 32), interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(save_img_filename, img)
        out_labels.append(labels[i])

    save_label_filename = os.path.join(args.output_dir, 'labels.txt')
    with open(save_label_filename, mode='w', encoding='utf-8') as f:
        for l in out_labels:
            f.write("{}\n".format(l))

    print("Converted img: %d" % len(out_labels))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='')
    args, _ = parser.parse_known_args()

    if not os.path.exists(args.input_dir):
        print("Input dir not exist")
        exit(-1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)

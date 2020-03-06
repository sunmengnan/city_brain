#!/usr/env/bin python3
import argparse
import os

import datetime

from libs.utils import check_dir_exist
from libs.config import load_config

OUTPUT_DIR = os.getenv("OUTPUT_DIR")
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def save_flags(args, save_dir):
    """
    Save flags into file, use date as file name
    :param args: tf.app.flags
    :param save_dir: dir to save flags file
    """
    filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".txt"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, filename)
    print("Save flags to %s" % filepath)

    cfg = load_config(args.cfg_name)

    with open(filepath, mode="w", encoding="utf-8") as f:
        d = vars(args)
        for k, v in d.items():
            f.write("%s: %s\n" % (k, v))

        print("=" * 30)
        for k, v in cfg.items():
            f.write('%s: %s\n' % (k, v))


def parse_infer_args(infer = True):
    if OUTPUT_DIR is None:
        output_dir = os.path.join(CURRENT_DIR, 'output')
    else:
        output_dir = OUTPUT_DIR

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', action='store_true', default=False)

    # 如果是使用 TFrecord 模式，则会加载相应数据目录下的所有 tf_record 文件
    # 如果是 JPG 模式，则会去加载相应目录下二级目录中的 jpg 图片 和 labels.txt
    #parser.add_argument('--train_dir', required=True)
    #parser.add_argument('--train_file_format', required=True, choices=['TF', 'JPG'])
    #parser.add_argument('--val_dir', required=True)
    #parser.add_argument('--val_file_format', required=True, choices=['TF', 'JPG'])
    #parser.add_argument('--test_dir', default=None, help='test 只支持小文件图片格式')

    #parser.add_argument('--restore', action='store_true', help='Whether to resotre checkpoint from ckpt_dir')
    #parser.add_argument('--restore_step', action='store_true', help='如果 restore step，lr 会减小')

    parser.add_argument('--tag', default='default', help='Subdirectory to create in checkpoint_dir/log_dir/result_dir')
    parser.add_argument('--ckpt_dir', default=os.path.join(output_dir, 'checkpoint'),
                        help='Directory to save tensorflow checkpoint')
    #parser.add_argument('--log_dir', default=os.path.join(output_dir, 'output/log'),
    #                    help='Directory to save tensorboard logs')
    parser.add_argument('--result_dir', default=os.path.join(output_dir, 'output/result'),
                        help='Directory to save val/test result')

    parser.add_argument('--chars_file',
                        default=os.path.join(CURRENT_DIR, 'data/ocr_chars/chn.txt'), help='Chars file to load')

    parser.add_argument('--cfg_name', default='raw', help="raw / squeeze/ dense / resnet / simple")

    parser.add_argument('--val_step', type=int, default=5000, help='Steps to do val.test and save checkpoint')
    parser.add_argument('--log_step', type=int, default=50, help='Steps save tensorboard summary')
    parser.add_argument('--display_step', type=int, default=10, help='Steps print loss to stdout')

    # Only for inference
    parser.add_argument('--infer_dir', default='./data/demo', help='Directory store infer images and labels')
    parser.add_argument('--infer_data_ordered', action='store_true', help='ground truth 存在 labels.txt 文件中')
    parser.add_argument('--load_sub_infer_dir', action='store_true', help='对 infer_dir 中的子目录进行测试')
    parser.add_argument('--infer_copy_failed', action='store_true', help='拷贝结果错误的测试数据图片到特定目录')
    parser.add_argument('--infer_batch_size', type=int, default=1)

    args, _ = parser.parse_known_args()

    #if (not infer) and (not os.path.exists(args.train_dir)):
    #    parser.error('train_dir not exist')

    #if (args.val_dir is not None) and (not os.path.exists(args.val_dir)):
    #    parser.error('val_dir not exist')

    #if (args.test_dir is not None) and (not os.path.exists(args.test_dir)):
    #    parser.error('test_dir not exist')

    if infer and (not os.path.exists(args.infer_dir)):
        parser.error('infer_dir not exist')

    args.ckpt_dir = os.path.join(args.ckpt_dir, args.tag)
    args.best_test_ckpt_dir = os.path.join(args.ckpt_dir, 'best_test_ckpt')
    args.flags_dir = os.path.join(args.ckpt_dir, "flags")
    #args.log_dir = os.path.join(args.log_dir, args.tag)
    args.result_dir = os.path.join(args.result_dir, args.tag)

    check_dir_exist(args.ckpt_dir)
    check_dir_exist(args.best_test_ckpt_dir)
    check_dir_exist(args.flags_dir)
    #check_dir_exist(args.log_dir)
    check_dir_exist(args.result_dir)

    save_flags(args, args.flags_dir)

    print('Use %s as base net' % args.cfg_name)

    return args


def parse_args(infer=False):
    if OUTPUT_DIR is None:
        output_dir = os.path.join(CURRENT_DIR, 'output')
    else:
        output_dir = OUTPUT_DIR

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', action='store_true', default=False)

    # 如果是使用 TFrecord 模式，则会加载相应数据目录下的所有 tf_record 文件
    # 如果是 JPG 模式，则会去加载相应目录下二级目录中的 jpg 图片 和 labels.txt
    parser.add_argument('--train_dir', required=True)
    parser.add_argument('--train_file_format', required=True, choices=['TF', 'JPG'])
    parser.add_argument('--val_dir', required=True)
    parser.add_argument('--val_file_format', required=True, choices=['TF', 'JPG'])
    parser.add_argument('--test_dir', default=None, help='test 只支持小文件图片格式')

    parser.add_argument('--restore', action='store_true', help='Whether to resotre checkpoint from ckpt_dir')
    parser.add_argument('--restore_step', action='store_true', help='如果 restore step，lr 会减小')

    parser.add_argument('--tag', default='default', help='Subdirectory to create in checkpoint_dir/log_dir/result_dir')
    parser.add_argument('--ckpt_dir', default=os.path.join(output_dir, 'checkpoint'),
                        help='Directory to save tensorflow checkpoint')
    parser.add_argument('--log_dir', default=os.path.join(output_dir, 'output/log'),
                        help='Directory to save tensorboard logs')
    parser.add_argument('--result_dir', default=os.path.join(output_dir, 'output/result'),
                        help='Directory to save val/test result')

    parser.add_argument('--chars_file',
                        default=os.path.join(CURRENT_DIR, 'data/ocr_chars/chn.txt'), help='Chars file to load')

    parser.add_argument('--cfg_name', default='raw', help="raw / squeeze/ dense / resnet / simple")

    parser.add_argument('--val_step', type=int, default=5000, help='Steps to do val.test and save checkpoint')
    parser.add_argument('--log_step', type=int, default=50, help='Steps save tensorboard summary')
    parser.add_argument('--display_step', type=int, default=10, help='Steps print loss to stdout')

    # Only for inference
    parser.add_argument('--infer_dir', default='./data/demo', help='Directory store infer images and labels')
    parser.add_argument('--infer_data_ordered', action='store_true', help='ground truth 存在 labels.txt 文件中')
    parser.add_argument('--load_sub_infer_dir', action='store_true', help='对 infer_dir 中的子目录进行测试')
    parser.add_argument('--infer_copy_failed', action='store_true', help='拷贝结果错误的测试数据图片到特定目录')
    parser.add_argument('--infer_batch_size', type=int, default=1)

    args, _ = parser.parse_known_args()

    if (not infer) and (not os.path.exists(args.train_dir)):
        parser.error('train_dir not exist')

    if (args.val_dir is not None) and (not os.path.exists(args.val_dir)):
        parser.error('val_dir not exist')

    if (args.test_dir is not None) and (not os.path.exists(args.test_dir)):
        parser.error('test_dir not exist')

    if infer and (not os.path.exists(args.infer_dir)):
        parser.error('infer_dir not exist')

    args.ckpt_dir = os.path.join(args.ckpt_dir, args.tag)
    args.best_test_ckpt_dir = os.path.join(args.ckpt_dir, 'best_test_ckpt')
    args.flags_dir = os.path.join(args.ckpt_dir, "flags")
    args.log_dir = os.path.join(args.log_dir, args.tag)
    args.result_dir = os.path.join(args.result_dir, args.tag)

    check_dir_exist(args.ckpt_dir)
    check_dir_exist(args.best_test_ckpt_dir)
    check_dir_exist(args.flags_dir)
    check_dir_exist(args.log_dir)
    check_dir_exist(args.result_dir)

    save_flags(args, args.flags_dir)

    print('Use %s as base net' % args.cfg_name)

    return args


if __name__ == '__main__':
    args = parse_args()

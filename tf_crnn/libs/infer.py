import time
import os
import math

import numpy as np

from libs import utils
from libs.img_dataset import ImgDataset
from nets.crnn import CRNN
from nets.cnn.paper_cnn import PaperCNN
import shutil


def calculate_accuracy(predicts, labels):
    """
    :param predicts: encoded predict result
    :param labels: ground true label
    :return: accuracy
    """
    assert len(predicts) == len(labels)

    correct_count = 0
    for i, p_label in enumerate(predicts):
        if p_label == labels[i]:
            correct_count += 1

    acc = correct_count / len(predicts)
    return acc, correct_count


def calculate_edit_distance_mean(edit_distences):
    """
    排除了 edit_distance == 0 的值计算编辑距离的均值
    :param edit_distences:
    :return:
    """
    data = np.array(edit_distences)
    data = data[data != 0]
    if len(data) == 0:
        return 0
    return np.mean(data)


def validation(sess, feeds, fetches, dataset, converter, result_dir, name,
               step=None, print_batch_info=False, copy_failed=False):
    """
    Save file name: {acc}_{step}.txt
    :param sess: tensorflow session
    :param model: crnn network
    :param result_dir:
    :param name: val, test, infer. used to create sub dir in result_dir
    :return:
    """
    sess.run(dataset.init_op)

    img_paths = []
    predicts = []
    trimed_predicts = []
    labels = []
    trimed_labels = []
    edit_distances = []
    total_batch_time = 0

    for batch in range(dataset.num_batches):
        img_batch, widths, label_batch, batch_labels, batch_img_paths = dataset.get_next_batch(sess)
        if len(batch_labels) == 0:
            continue

        batch_start_time = time.time()

        feed = {feeds['inputs']: img_batch,
                feeds['labels']: label_batch,
                feeds['sequence_length']: PaperCNN.get_sequence_lengths(widths),
                feeds['is_training']: False}

        try:
            batch_predicts, edit_distance, batch_edit_distances = sess.run(fetches, feed)
        except Exception:
            print(batch_labels)
            continue
        batch_predicts = [converter.decode(p, CRNN.CTC_INVALID_INDEX) for p in batch_predicts]

        trimed_batch_predicts = [utils.remove_all_symbols(txt) for txt in batch_predicts]
        trimed_batch_labels = [utils.remove_all_symbols(txt) for txt in batch_labels]

        img_paths.extend(batch_img_paths)
        predicts.extend(batch_predicts)
        labels.extend(batch_labels)
        trimed_predicts.extend(trimed_batch_predicts)
        trimed_labels.extend(trimed_batch_labels)
        edit_distances.extend(batch_edit_distances)

        acc, correct_count = calculate_accuracy(batch_predicts, batch_labels)
        trimed_acc, trimed_correct_count = calculate_accuracy(trimed_batch_predicts, trimed_batch_labels)

        batch_time = time.time() - batch_start_time
        total_batch_time += batch_time
        if print_batch_info:
            print("{:.03f}s [{}/{}] acc: {:.03f}({}/{}), edit_distance: {:.03f}, trim_acc {:.03f}({}/{})"
                  .format(batch_time, batch, dataset.num_batches,
                          acc, correct_count, dataset.batch_size,
                          edit_distance,
                          trimed_acc, trimed_correct_count, dataset.batch_size))

    acc, correct_count = calculate_accuracy(predicts, labels)
    trimed_acc, trimed_correct_count = calculate_accuracy(trimed_predicts, trimed_labels)
    edit_distance_mean = calculate_edit_distance_mean(edit_distances)
    total_edit_distance = sum(edit_distances)

    acc_str = "Accuracy: {:.03f} ({}/{}), Trimed Accuracy: {:.03f} ({}/{})" \
              "Total edit distance: {:.03f}, " \
              "Average edit distance: {:.03f}, Average batch time: {:.03f}" \
        .format(acc, correct_count, dataset.size,
                trimed_acc, trimed_correct_count, dataset.size,
                total_edit_distance, edit_distance_mean, total_batch_time / dataset.num_batches)

    print(acc_str)

    save_dir = os.path.join(result_dir, name)
    utils.check_dir_exist(save_dir)

    result_file_path = save_txt_result(save_dir, acc, step, labels, predicts, 'acc',
                                       edit_distances, acc_str)

    save_txt_result(save_dir, acc, step, labels, predicts, 'acc', edit_distances,
                    acc_str, only_failed=True)

    save_txt_result(save_dir, trimed_acc, step, trimed_labels, trimed_predicts, 'tacc',
                    edit_distances)

    save_txt_result(save_dir, trimed_acc, step, trimed_labels, trimed_predicts, 'tacc',
                    edit_distances, only_failed=True)

    save_txt_4_analyze(save_dir, labels, predicts, 'acc', step)
    save_txt_4_analyze(save_dir, trimed_labels, trimed_predicts, 'tacc', step)

    # Copy image not all match to a dir
    # TODO: we will only save failed imgs for acc
    if copy_failed:
        failed_infer_img_dir = result_file_path[:-4] + "_failed"
        if os.path.exists(failed_infer_img_dir) and os.path.isdir(failed_infer_img_dir):
            shutil.rmtree(failed_infer_img_dir)

        utils.check_dir_exist(failed_infer_img_dir)

        failed_image_indices = []
        for i, val in enumerate(edit_distances):
            if val != 0:
                failed_image_indices.append(i)

        for i in failed_image_indices:
            img_path = img_paths[i]
            img_name = img_path.split("/")[-1]
            dst_path = os.path.join(failed_infer_img_dir, img_name)
            shutil.copyfile(img_path, dst_path)

        failed_infer_result_file_path = os.path.join(failed_infer_img_dir, "result.txt")
        with open(failed_infer_result_file_path, 'w', encoding='utf-8') as f:
            for i in failed_image_indices:
                p_label = predicts[i]
                t_label = labels[i]
                f.write("{}\n".format(img_paths[i]))
                f.write("input:   {:17s} length: {}\n".format(t_label, len(t_label)))
                f.write("predict: {:17s} length: {}\n".format(p_label, len(p_label)))
                f.write("edit distance:  {}\n".format(edit_distances[i]))
                f.write('-' * 30 + '\n')

    return acc, trimed_acc, edit_distance_mean, total_edit_distance, correct_count, trimed_correct_count


def save_txt_4_analyze(save_dir, labels, predicts, acc_type, step):
    """
    把测试集的真值和预测结果放在保存在同一个 txt 文件中，方便统计
    """
    txt_path = os.path.join(save_dir, '%d_%s_gt_and_pred.txt' % (step, acc_type))

    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, p_label in enumerate(predicts):
            t_label = labels[i]
            f.write("{}__$__{}\n".format(t_label, p_label))


def save_txt_result(save_dir, acc, step, labels, predicts, acc_type,
                    edit_distances=None, acc_str=None, only_failed=False):
    """
    :param acc_type:  'acc' or 'tacc'
    :return:
    """
    failed_suffix = ''
    if only_failed:
        failed_suffix = 'failed'

    if step is not None:
        txt_path = os.path.join(save_dir, '%d_%s_%.3f_%s.txt' % (step, acc_type, acc, failed_suffix))
    else:
        txt_path = os.path.join(save_dir, '%s_%.3f_%s.txt' % (acc_type, acc, failed_suffix))

    print("Write result to %s" % txt_path)
    with open(txt_path, 'w', encoding='utf-8') as f:
        for i, p_label in enumerate(predicts):
            t_label = labels[i]
            all_match = (t_label == p_label)

            if only_failed and all_match:
                continue

            # f.write("{}\n".format(img_paths[i]))
            f.write("input:   {:17s} length: {}\n".format(t_label, len(t_label)))
            f.write("predict: {:17s} length: {}\n".format(p_label, len(p_label)))
            f.write("all match:  {}\n".format(1 if all_match else 0))

            if edit_distances:
                f.write("edit distance:  {}\n".format(edit_distances[i]))

            f.write('-' * 30 + '\n')

        if acc_str:
            f.write(acc_str + "\n")

    return txt_path

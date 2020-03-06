import os
import time
import math

from libs.config import load_config

# RNG_SEED = 42
# import numpy as np
# np.random.seed(RNG_SEED)

import tensorflow as tf
# tf.set_random_seed(RNG_SEED)

import libs.utils as utils
import libs.tf_utils as tf_utils
from libs.TFRecordDataset import TFRecordDataset
from libs.img_dataset import ImgDataset
from libs.label_converter import LabelConverter
import libs.infer as infer

from nets.crnn import CRNN
from nets.cnn.paper_cnn import PaperCNN
from parse_args import parse_args

from libs.checkmate import BestCheckpointSaver
from libs.data_augumenter import DataAugumenter


def create_ds(img_dir, file_format, batch_size, converter, da):
    if file_format == 'TF':
        return TFRecordDataset(img_dir,
                               converter=converter,
                               batch_size=batch_size,
                               da=da)
    elif file_format == 'JPG':
        return ImgDataset(img_dir,
                          converter=converter,
                          batch_size=batch_size,
                          data_ordered = True,
                          da=da)


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.cfg = load_config(args.cfg_name)

        self.converter = LabelConverter(args.chars_file)

        self.da = DataAugumenter(self.cfg)

        self.tr_ds = create_ds(args.train_dir, args.train_file_format, self.cfg.batch_size, self.converter, self.da)
        print("Training image num: %d" % self.tr_ds.size)
        self.val_ds = create_ds(args.val_dir, args.val_file_format, self.cfg.batch_size, self.converter, self.da)

        self.cfg.lr_boundaries = [self.tr_ds.num_batches * epoch for epoch in self.cfg.lr_decay_epochs]
        self.cfg.lr_values = [self.cfg.lr * (self.cfg.lr_decay_rate ** i) for i in
                              range(len(self.cfg.lr_boundaries) + 1)]

        self.model = CRNN(self.cfg, num_classes=self.converter.num_classes)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.epoch_start_index = 0
        self.batch_start_index = 0
        self.global_step = 0

        self.best_test_ckpt_saver = BestCheckpointSaver(
            save_dir=self.args.best_test_ckpt_dir,
            num_to_keep=2,
            maximize=True
        )

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=8)
        self.train_writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)

        if self.args.restore:
            self._restore()

        print('Begin training...')
        for epoch in range(self.epoch_start_index, self.cfg.epochs):
            self.sess.run(self.tr_ds.init_op)

            for batch in range(self.batch_start_index, self.tr_ds.num_batches):
                batch_start_time = time.time()

                try:
                    if batch != 0 and (batch % self.args.log_step == 0):
                        batch_cost, reg_loss, global_step, lr = self._train_with_summary()
                    else:
                        batch_cost, reg_loss, global_step, lr = self._train()
                except Exception as e:
                    print("Error train step: {}".format(e))
                    continue

                self.global_step = global_step

                if global_step % self.args.display_step == 0:
                    print(
                        "epoch: {}, batch: {}/{}, step: {}, time: {:.02f}s, loss: {:.03}, reg_loss: {:.03}, lr: {:.05}"
                            .format(epoch, batch, self.tr_ds.num_batches,
                                    global_step, time.time() - batch_start_time,
                                    batch_cost, reg_loss, lr))

                if global_step != 0 and (global_step % self.args.val_step == 0):
                    val_acc, _, _ = self._do_val(self.val_ds, epoch, global_step, "val")

                    test_acc, trimed_test_acc = self._do_test(epoch, global_step, self.args.test_dir)

                    self._save_checkpoint(self.args.ckpt_dir, global_step, val_acc, test_acc, trimed_test_acc)

            self.batch_start_index = 0

    def _restore(self):
        utils.restore_ckpt(self.sess, self.saver, self.args.ckpt_dir)

        if self.args.restore_step:
            step_restored = self.sess.run(self.model.global_step)
            self.global_step = step_restored
        else:
            step_restored = 0
            assign_global_step_zero = tf.assign(self.model.global_step, 0)
            self.sess.run(assign_global_step_zero)

        self.epoch_start_index = math.floor(step_restored / self.tr_ds.num_batches)
        self.batch_start_index = step_restored % self.tr_ds.num_batches

        print("Restored global step: %d" % step_restored)
        print("Restored epoch: %d" % self.epoch_start_index)
        print("Restored batch_start_index: %d" % self.batch_start_index)

    def _train(self):
        img_batch, widths, label_batch, labels, _ = self.tr_ds.get_next_batch(self.sess)

        feed = {self.model.inputs: img_batch,
                self.model.labels: label_batch,
                self.model.seq_len: PaperCNN.get_sequence_lengths(widths),
                self.model.is_training: True}

        fetches = [self.model.total_loss,
                   self.model.ctc_loss,
                   self.model.global_step,
                   self.model.lr,
                   self.model.train_op]

        reg_loss = 0.0
        if self.model.regularization_loss is None:
            batch_cost, _, global_step, lr, _ = self.sess.run(fetches, feed)
        else:
            fetches.append(self.model.regularization_loss)
            batch_cost, _, global_step, lr, _, reg_loss = self.sess.run(fetches, feed)

        return batch_cost, reg_loss, global_step, lr

    def _train_with_summary(self):
        img_batch, widths, label_batch, labels, _ = self.tr_ds.get_next_batch(self.sess)
        feed = {self.model.inputs: img_batch,
                self.model.labels: label_batch,
                self.model.seq_len: PaperCNN.get_sequence_lengths(widths),
                self.model.is_training: True}

        fetches = [self.model.total_loss,
                   self.model.ctc_loss,
                   self.model.global_step,
                   self.model.lr,
                   self.model.merged_summay,
                   self.model.dense_decoded,
                   self.model.edit_distance,
                   self.model.train_op]

        reg_loss = 0.0
        if self.model.regularization_loss is None:
            batch_cost, _, global_step, lr, summary, predicts, edit_distance, _ = self.sess.run(fetches, feed)
        else:
            fetches.append(self.model.regularization_loss)
            batch_cost, _, global_step, lr, summary, predicts, edit_distance, _, reg_loss = self.sess.run(fetches, feed)

        self.train_writer.add_summary(summary, global_step)

        predicts = [self.converter.decode(p, CRNN.CTC_INVALID_INDEX) for p in predicts]
        accuracy, _ = infer.calculate_accuracy(predicts, labels)

        tf_utils.add_scalar_summary(self.train_writer, "train_accuracy", accuracy, global_step)
        tf_utils.add_scalar_summary(self.train_writer, "train_edit_distance", edit_distance, global_step)

        return batch_cost, reg_loss, global_step, lr

    def _do_val(self, dataset, epoch, step, name):
        if dataset is None:
            return None

        acc, trimed_acc, edit_distance, total_edit_distance, correct_count, trimed_correct_count = infer.validation(
            self.sess, self.model.feeds(),
            self.model.fetches(),
            dataset, self.converter,
            self.args.result_dir,
            name, step)

        tf_utils.add_scalar_summary(self.train_writer, "%s_trimed_accuracy" % name, trimed_acc, step)
        tf_utils.add_scalar_summary(self.train_writer, "%s_accuracy" % name, acc, step)
        tf_utils.add_scalar_summary(self.train_writer, "%s_edit_distance" % name, edit_distance, step)
        tf_utils.add_scalar_summary(self.train_writer, "%s_total_edit_distance" % name, total_edit_distance, step)

        print("epoch: %d/%d, %s accuracy = %.3f" % (epoch, self.cfg.epochs, name, acc))
        return acc, correct_count, trimed_correct_count

    def _do_test(self, epoch, step, test_dir):
        if test_dir is None:
            return None, None

        total_count = 0
        total_correct_count = 0
        total_trimed_correct_count = 0
        ds = ImgDataset(test_dir, self.converter, batch_size=1, shuffle=False)

        acc, correct_count, trimed_correct_count = self._do_val(ds, epoch, step, 'test/' + ds.img_dirname)
        total_count += ds.size
        total_correct_count += correct_count
        total_trimed_correct_count += trimed_correct_count

        total_acc = total_correct_count / total_count
        total_trimed_acc = total_trimed_correct_count / total_count

        tf_utils.add_scalar_summary(self.train_writer, "test_accuracy", total_acc, step)
        tf_utils.add_scalar_summary(self.train_writer, "trimed_test_accuracy", total_trimed_acc, step)

        return total_acc, total_trimed_acc

    def _save_checkpoint(self, ckpt_dir, step, val_acc=None, test_acc=None, trimed_test_acc=None):
        ckpt_name = "crnn_%d" % step
        if val_acc is not None:
            ckpt_name += '_val_%.03f' % val_acc
        if test_acc is not None:
            ckpt_name += '_test_%.03f' % test_acc
        if trimed_test_acc is not None:
            ckpt_name += '_trimed_test_%.03f' % trimed_test_acc

        name = os.path.join(ckpt_dir, ckpt_name)
        print("save checkpoint %s" % name)

        meta_exists, meta_file_name = self._meta_file_exist(ckpt_dir)

        self.saver.save(self.sess, name)

        if test_acc is not None:
            self.best_test_ckpt_saver.handle(test_acc, self.sess, step)

        # remove old meta file to save disk space
        if meta_exists:
            try:
                os.remove(os.path.join(ckpt_dir, meta_file_name))
            except:
                print('Remove meta file failed: %s' % meta_file_name)

    def _meta_file_exist(self, ckpt_dir):
        fnames = os.listdir(ckpt_dir)
        meta_exists = False
        meta_file_name = ''
        for n in fnames:
            if 'meta' in n:
                meta_exists = True
                meta_file_name = n
                break

        return meta_exists, meta_file_name


def main():
    dev = '/gpu:0'
    args = parse_args()
    with tf.device(dev):
        trainer = Trainer(args)
        trainer.train()


if __name__ == '__main__':
    main()

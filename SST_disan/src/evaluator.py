from configs import cfg
from src.utils.record_log import _logger
import numpy as np
import tensorflow as tf
from src.analysis import OutputAnalysis
import os, shutil

class Evaluator(object):
    def __init__(self, model):
        self.model = model
        self.global_step = model.global_step

        ## ---- summary----
        self.build_summary()
        self.writer = tf.summary.FileWriter(cfg.summary_dir)

    # --- external use ---
    def get_evaluation(self, sess, dataset_obj, global_step=None):
        _logger.add()
        _logger.add('getting evaluation result for %s' % dataset_obj.data_type)

        logits_list, loss_list, accu_list = [], [], []
        is_sent_list = []
        for sample_batch, _, _, _ in dataset_obj.generate_batch_sample_iter():
            feed_dict = self.model.get_feed_dict(sample_batch, 'dev')
            logits, loss, accu = sess.run([self.model.logits,
                                           self.model.loss, self.model.accuracy], feed_dict)
            logits_list.append(np.argmax(logits, -1))
            loss_list.append(loss)
            accu_list.append(accu)
            is_sent_list += [sample['is_sent'] for sample in sample_batch]
        logits_array = np.concatenate(logits_list, 0)
        loss_value = np.mean(loss_list)
        accu_array = np.concatenate(accu_list, 0)
        accu_value =np.mean(accu_array)
        sent_accu_list = []
        for idx, is_sent in enumerate(is_sent_list):
            if is_sent:
                sent_accu_list.append(accu_array[idx])
        sent_accu_value = np.mean(sent_accu_list)

        # analysis
        # analysis_save_dir = cfg.mkdir(cfg.answer_dir,'gs_%s'%global_step or 'test')
        # OutputAnalysis.do_analysis(dataset_obj, logits_array, accu_array, analysis_save_dir,
        #                            cfg.fine_grained)

        # add summary
        if global_step is not None:
            if dataset_obj.data_type == 'train':
                summary_feed_dict = {
                    self.train_loss: loss_value,
                    self.train_accuracy: accu_value,
                    self.train_sent_accuracy: sent_accu_value,
                }
                summary = sess.run(self.train_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            elif dataset_obj.data_type == 'dev':
                summary_feed_dict = {
                    self.dev_loss: loss_value,
                    self.dev_accuracy: accu_value,
                    self.dev_sent_accuracy: sent_accu_value,
                }
                summary = sess.run(self.dev_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            else:
                summary_feed_dict = {
                    self.test_loss: loss_value,
                    self.test_accuracy: accu_value,
                    self.test_sent_accuracy: sent_accu_value,
                }
                summary = sess.run(self.test_summaries, summary_feed_dict)
                self.writer.add_summary(summary, global_step)
            return loss_value, accu_value, sent_accu_value

    def get_evaluation_file_output(self, sess, dataset_obj, global_step, deleted_step):
        _logger.add()
        _logger.add('get evaluation file output for %s' % dataset_obj.data_type)
        # delete old file
        if deleted_step is not None:
            delete_name = 'gs_%d' % deleted_step
            delete_path = os.path.join(cfg.answer_dir, delete_name)
            if os.path.exists(delete_path):
                shutil.rmtree(delete_path)
            _logger.add()
            _logger.add('getting evaluation result for %s' % dataset_obj.data_type)

        logits_list, loss_list, accu_list = [], [], []
        is_sent_list = []
        for sample_batch, _, _, _ in dataset_obj.generate_batch_sample_iter():
            feed_dict = self.model.get_feed_dict(sample_batch, 'dev')
            logits, loss, accu = sess.run([self.model.logits,
                                           self.model.loss, self.model.accuracy], feed_dict)
            logits_list.append(np.argmax(logits, -1))
            loss_list.append(loss)
            accu_list.append(accu)
            is_sent_list += [sample['is_sent'] for sample in sample_batch]
        logits_array = np.concatenate(logits_list, 0)
        loss_value = np.mean(loss_list)
        accu_array = np.concatenate(accu_list, 0)
        accu_value = np.mean(accu_array)
        sent_accu_list = []
        for idx, is_sent in enumerate(is_sent_list):
            if is_sent:
                sent_accu_list.append(accu_array[idx])
        sent_accu_value = np.mean(sent_accu_list)

        # analysis
        analysis_save_dir = cfg.mkdir(cfg.answer_dir,'gs_%s'%global_step or 'test')
        OutputAnalysis.do_analysis(dataset_obj, logits_array, accu_array, analysis_save_dir,
                                   cfg.fine_grained)


    # --- internal use ------
    def build_summary(self):
        with tf.name_scope('train_summaries'):
            self.train_loss = tf.placeholder(tf.float32, [], 'train_loss')
            self.train_accuracy = tf.placeholder(tf.float32, [], 'train_accuracy')
            self.train_sent_accuracy = tf.placeholder(tf.float32, [], 'train_sent_accuracy')
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_loss', self.train_loss))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_accuracy', self.train_accuracy))
            tf.add_to_collection('train_summaries_collection', tf.summary.scalar('train_sent_accuracy',
                                                                                 self.train_sent_accuracy))
            self.train_summaries = tf.summary.merge_all('train_summaries_collection')

        with tf.name_scope('dev_summaries'):
            self.dev_loss = tf.placeholder(tf.float32, [], 'dev_loss')
            self.dev_accuracy = tf.placeholder(tf.float32, [], 'dev_accuracy')
            self.dev_sent_accuracy = tf.placeholder(tf.float32, [], 'dev_sent_accuracy')
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_loss',self.dev_loss))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_accuracy',self.dev_accuracy))
            tf.add_to_collection('dev_summaries_collection', tf.summary.scalar('dev_sent_accuracy',
                                                                               self.dev_sent_accuracy))
            self.dev_summaries = tf.summary.merge_all('dev_summaries_collection')

        with tf.name_scope('test_summaries'):
            self.test_loss = tf.placeholder(tf.float32, [], 'test_loss')
            self.test_accuracy = tf.placeholder(tf.float32, [], 'test_accuracy')
            self.test_sent_accuracy = tf.placeholder(tf.float32, [], 'test_sent_accuracy')
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_loss',self.test_loss))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_accuracy',self.test_accuracy))
            tf.add_to_collection('test_summaries_collection', tf.summary.scalar('test_sent_accuracy',
                                                                                self.test_sent_accuracy))
            self.test_summaries = tf.summary.merge_all('test_summaries_collection')




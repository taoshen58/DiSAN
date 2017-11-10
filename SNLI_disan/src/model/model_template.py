from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod

class ModelTemplate(metaclass=ABCMeta):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        self.scope = scope
        self.global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                           initializer=tf.constant_initializer(0), trainable=False)


        self.token_emb_mat, self.glove_emb_mat = token_emb_mat, glove_emb_mat

        # ---- place holder -----
        self.sent1_token = tf.placeholder(tf.int32, [None, None], name='sent1_token')
        self.sent1_char = tf.placeholder(tf.int32, [None, None, tl], name='sent1_char')

        self.sent2_token = tf.placeholder(tf.int32, [None, None], name='sent2_token')
        self.sent2_char = tf.placeholder(tf.int32, [None, None, tl], name='sent2_char')

        self.gold_label = tf.placeholder(tf.int32, [None], name='gold_label')
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')

        # tree
        self.s1_op_list = tf.placeholder(tf.int32, [None, None], name='s1_op_list')  # bs,ol1
        self.s1_reduce_mat = tf.placeholder(tf.int32, [None, None, None], name='s1_reduce_mat')  # bs,ol1,mc
        self.s1_stack_mask = tf.not_equal(self.s1_op_list, tf.zeros_like(self.s1_op_list))

        self.s2_op_list = tf.placeholder(tf.int32, [None, None], name='s2_op_list')   # bs,ol2
        self.s2_reduce_mat = tf.placeholder(tf.int32, [None, None, None], name='s2_reduce_mat')  # bs,ol2,mc
        self.s2_stack_mask = tf.not_equal(self.s2_op_list, tf.zeros_like(self.s2_op_list))

        # ----------- parameters -------------
        self.tds, self.cds = tds, cds
        self.tl = tl
        self.tel = cfg.word_embedding_length
        self.cel = cfg.char_embedding_length
        self.cos = cfg.char_out_size
        self.ocd = list(map(int, cfg.out_channel_dims.split(',')))
        self.fh = list(map(int, cfg.filter_heights.split(',')))
        self.hn = cfg.hidden_units_num
        self.finetune_emb = cfg.fine_tune

        self.output_class = 3 # 0 for contradiction, 1 for neural and 2 for entailment
        self.bs = tf.shape(self.sent1_token)[0]
        self.sl1 = tf.shape(self.sent1_token)[1]
        self.sl2 = tf.shape(self.sent2_token)[1]

        self.ol1 = tf.shape(self.s1_op_list)[1]
        self.ol2 = tf.shape(self.s2_op_list)[1]

        self.mc = tf.shape(self.s1_reduce_mat)[2]

        # ------------ other ---------
        self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
        self.sent1_char_mask = tf.cast(self.sent1_char, tf.bool)
        self.sent1_token_len = tf.reduce_sum(tf.cast(self.sent1_token_mask, tf.int32), -1)
        self.sent1_char_len = tf.reduce_sum(tf.cast(self.sent1_char_mask, tf.int32), -1)

        self.sent2_token_mask = tf.cast(self.sent2_token, tf.bool)
        self.sent2_char_mask = tf.cast(self.sent2_char, tf.bool)
        self.sent2_token_len = tf.reduce_sum(tf.cast(self.sent2_token_mask, tf.int32), -1)
        self.sent2_char_len = tf.reduce_sum(tf.cast(self.sent2_char_mask, tf.int32), -1)

        self.tensor_dict = {}

        # ---------------- for dynamic learning rate -------------------
        self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')
        self.learning_rate_value = cfg.learning_rate
        self.previous_dev_loss = []

        for i in reversed(range(3)):
            self.previous_dev_loss.append((i+1) * 1e5)
        self.learning_rate_updated = False

        # ------ start ------
        self.logits = None
        self.loss = None
        self.accuracy = None
        self.var_ema = None
        self.ema = None
        self.summary = None
        self.opt = None
        self.train_op = None

    @abstractmethod
    def build_network(self):
        pass

    def build_loss(self):
        # weight_decay
        with tf.name_scope("weight_decay"):
            for var in set(tf.get_collection('reg_vars', self.scope)):
                weight_decay = tf.multiply(tf.nn.l2_loss(var), cfg.wd,
                                           name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                tf.add_to_collection('losses', weight_decay)
        reg_vars = tf.get_collection('losses', self.scope)
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        _logger.add('regularization var num: %d' % len(reg_vars))
        _logger.add('trainable var num: %d' % len(trainable_vars))
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.gold_label,
            logits=self.logits
        )
        tf.add_to_collection('losses', tf.reduce_mean(losses, name='xentropy_loss_mean'))
        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        tf.add_to_collection('ema/scalar', loss)
        return loss

    def build_accuracy(self):
        correct = tf.equal(
            tf.cast(tf.argmax(self.logits, -1), tf.int32),
            self.gold_label
        )  # [bs]
        return tf.cast(correct, tf.float32)

    def update_tensor_add_ema_and_opt(self):
        self.logits = self.build_network()
        self.loss = self.build_loss()
        self.accuracy = self.build_accuracy()

        # ------------ema-------------
        if True:
            self.var_ema = tf.train.ExponentialMovingAverage(cfg.var_decay)
            self.build_var_ema()

        if cfg.mode == 'train':
            self.ema = tf.train.ExponentialMovingAverage(cfg.decay)
            self.build_ema()
        self.summary = tf.summary.merge_all()

        # ---------- optimization ---------
        if cfg.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif cfg.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
        elif cfg.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            raise AttributeError('no optimizer named as \'%s\'' % cfg.optimizer)


        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        # trainable param num:
        # print params num
        all_params_num = 0
        for elem in trainable_vars:
            # elem.name
            var_name = elem.name.split(':')[0]
            if var_name.endswith('emb_mat'): continue
            params_num = 1
            for l in elem.get_shape().as_list(): params_num *= l
            all_params_num += params_num
        _logger.add('Trainable Parameters Number: %d' % all_params_num)

        self.train_op = self.opt.minimize(self.loss, self.global_step,
                                          var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope))

    def build_var_ema(self):
        ema_op = self.var_ema.apply(tf.trainable_variables(),)
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def build_ema(self):
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + \
                  tf.get_collection("ema/vector", scope=self.scope)
        ema_op = self.ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = self.ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def step(self, sess, batch_samples, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, 'train')

        # out_tensor_dict_1= sess.run(self.tensor_dict, feed_dict=feed_dict)
        cfg.time_counter.add_start()
        if get_summary:
            loss, summary, train_op = sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)

        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        cfg.time_counter.add_stop()

        # out_tensor_dict_2 = sess.run(self.tensor_dict, feed_dict=feed_dict)
        # anchor = 0

        return loss, summary, train_op

    def update_learning_rate(self, current_dev_loss, global_step, lr_decay_factor=0.7):
        if cfg.dy_lr:
            method = 1
            if method == 0:
                assert len(self.previous_dev_loss) >= 1
                self.previous_dev_loss.append(current_dev_loss)
                delta = []
                pre_loss = self.previous_dev_loss.pop(0)
                for loss in self.previous_dev_loss:
                    delta.append(pre_loss < loss)
                    pre_loss = loss

                do_decay = True
                for d in delta:
                    do_decay = do_decay and d

                if do_decay:
                    self.learning_rate_value *= lr_decay_factor
                    self.learning_rate_updated = True
                    _logger.add('found dev loss increases, decrease learning rate to: %f' % self.learning_rate_value)

                if global_step % 10000 == 0:
                    if not self.learning_rate_updated:
                        self.learning_rate_value *= 0.9
                        _logger.add('decrease learning rate to: %f' % self.learning_rate_value)
                    else:
                        self.learning_rate_updated = False
            elif method == 1:
                if self.learning_rate_value < 5e-6:
                    return
                if global_step % 5000 == 0:
                    self.learning_rate_value *= lr_decay_factor





    def get_feed_dict(self, sample_batch, data_type='train'):
        # max lens
        sl1, sl2 = 0, 0

        for sample in sample_batch:
            sl1 = max(sl1, len(sample['sentence1_token_digital']))
            sl2 = max(sl2, len(sample['sentence2_token_digital']))


        # token and char
        sent1_token_b = []
        sent1_char_b = []
        sent2_token_b = []
        sent2_char_b = []
        for sample in sample_batch:
            sent1_token = np.zeros([sl1], cfg.intX)
            sent1_char = np.zeros([sl1, self.tl], cfg.intX)
            for idx_t, (token, char_seq_v) in enumerate(zip(sample['sentence1_token_digital'],
                                                            sample['sentence1_char_digital'])):
                sent1_token[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c < self.tl:
                        sent1_char[idx_t, idx_c] = char

            sent2_token = np.zeros([sl2], cfg.intX)
            sent2_char = np.zeros([sl2, self.tl], cfg.intX)

            for idx_t, (token, char_seq_v) in enumerate(zip(sample['sentence2_token_digital'],
                                                            sample['sentence2_char_digital'])):
                sent2_token[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c < self.tl:
                        sent2_char[idx_t, idx_c] = char
            sent1_token_b.append(sent1_token)
            sent1_char_b.append(sent1_char)
            sent2_token_b.append(sent2_token)
            sent2_char_b.append(sent2_char)
        sent1_token_b = np.stack(sent1_token_b)
        sent1_char_b = np.stack(sent1_char_b)
        sent2_token_b = np.stack(sent2_token_b)
        sent2_char_b = np.stack(sent2_char_b)

        # label
        gold_label_b = []
        for sample in sample_batch:
            gold_label_int = None
            if sample['gold_label'] == 'contradiction':
                gold_label_int = 0
            elif sample['gold_label'] == 'neutral':
                gold_label_int = 1
            elif sample['gold_label'] == 'entailment':
                gold_label_int = 2
            assert gold_label_int is not None
            gold_label_b.append(gold_label_int)
        gold_label_b = np.stack(gold_label_b).astype(cfg.intX)

        feed_dict = {
            self.sent1_token: sent1_token_b, self.sent1_char: sent1_char_b,
            self.sent2_token: sent2_token_b, self.sent2_char: sent2_char_b,
            self.gold_label:gold_label_b,
            self.is_train: True if data_type == 'train' else False,
            self.learning_rate: self.learning_rate_value,
        }

        # tree
        if not cfg.data_clip_method == 'no_tree':
            ol1, ol2 = 0, 0
            mc = 0
            for sample in sample_batch:
                ol1 = max(ol1, len(sample['sentence1_binary_parse_node_list']))
                ol2 = max(ol2, len(sample['sentence2_binary_parse_node_list']))
                mc = max(
                    [mc] + [len(red) for red in
                            (sample['s1_sr_info']['reduce_mat'] + sample['s2_sr_info']['reduce_mat'])]
                )
            s1_op_list_b = []
            s1_reduce_mat_b = []
            s2_op_list_b = []
            s2_reduce_mat_b = []
            for sample in sample_batch:
                # sent1
                s1_op_list = np.zeros([ol1], cfg.intX)
                s1_reduce_mat = np.zeros([ol1, mc], cfg.intX) * -1
                for idx_o, (ol, red) in enumerate(zip(sample['s1_sr_info']['op_list'],
                                                      sample['s1_sr_info']['reduce_mat'])):
                    s1_op_list[idx_o] = ol
                    for idx_r, r in enumerate(red):
                        s1_reduce_mat[idx_o, idx_r] = r
                s1_op_list_b.append(s1_op_list)
                s1_reduce_mat_b.append(s1_reduce_mat)

                # sent2
                s2_op_list = np.zeros([ol2], cfg.intX)
                s2_reduce_mat = np.zeros([ol2, mc], cfg.intX) * -1
                for idx_o, (ol, red) in enumerate(zip(sample['s2_sr_info']['op_list'],
                                                      sample['s2_sr_info']['reduce_mat'])):
                    s2_op_list[idx_o] = ol
                    for idx_r, r in enumerate(red):
                        s2_reduce_mat[idx_o, idx_r] = r
                s2_op_list_b.append(s2_op_list)
                s2_reduce_mat_b.append(s2_reduce_mat)

            s1_op_list_b = np.stack(s1_op_list_b)
            s1_reduce_mat_b = np.stack(s1_reduce_mat_b)
            s2_op_list_b = np.stack(s2_op_list_b)
            s2_reduce_mat_b = np.stack(s2_reduce_mat_b)

            extra_feed_dict = {self.s1_op_list: s1_op_list_b, self.s1_reduce_mat: s1_reduce_mat_b,
                               self.s2_op_list: s2_op_list_b, self.s2_reduce_mat: s2_reduce_mat_b}
            feed_dict.update(extra_feed_dict)

        return feed_dict











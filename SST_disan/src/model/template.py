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
        self.token_seq = tf.placeholder(tf.int32, [None, None], name='token_seq')
        self.char_seq = tf.placeholder(tf.int32, [None, None, tl], name='context_char')

        self.op_list = tf.placeholder(tf.int32, [None, None], name='op_lists')  # bs,sol
        self.reduce_mat = tf.placeholder(tf.int32, [None, None, None], name='reduce_mats')  # [bs,sol,mc]

        self.sentiment_label = tf.placeholder(tf.int32, [None], name='sentiment_label')  # bs
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')


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

        self.output_class = 5 if cfg.fine_grained else 2

        self.bs = tf.shape(self.token_seq)[0]
        self.sl = tf.shape(self.token_seq)[1]
        self.ol = tf.shape(self.op_list)[1]
        self.mc = tf.shape(self.reduce_mat)[2]

        # ------------ other ---------
        self.token_mask = tf.cast(self.token_seq, tf.bool)
        self.char_mask = tf.cast(self.char_seq, tf.bool)
        self.token_len = tf.reduce_sum(tf.cast(self.token_mask, tf.int32), -1)
        self.char_len = tf.reduce_sum(tf.cast(self.char_mask, tf.int32), -1)

        self.stack_mask = tf.not_equal(self.op_list, tf.zeros_like(self.op_list))

        self.tensor_dict = {}

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
            labels=self.sentiment_label,
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
            self.sentiment_label
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
            assert cfg.learning_rate > 0.1 and cfg.learning_rate < 1.
            self.opt = tf.train.AdadeltaOptimizer(cfg.learning_rate)
        elif cfg.optimizer.lower() == 'adam':
            assert cfg.learning_rate < 0.1
            self.opt = tf.train.AdamOptimizer(cfg.learning_rate)
        elif cfg.optimizer.lower() == 'rmsprop':
            assert cfg.learning_rate < 0.1
            self.opt = tf.train.RMSPropOptimizer(cfg.learning_rate)
        else:
            raise AttributeError('no optimizer named as \'%s\'' % cfg.optimizer)


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

    def get_feed_dict(self, sample_batch, data_type='train'):
        # max lens
        sl, ol, mc = 0, 0, 0
        for sample in sample_batch:
            sl = max(sl, len(sample['root_node']['token_seq']))
            ol = max(ol, len(sample['shift_reduce_info']['op_list']))
            for reduce_list in sample['shift_reduce_info']['reduce_mat']:
                mc = max(mc, len(reduce_list))

        assert mc == 0 or mc == 2, mc

        # token and char
        token_seq_b = []
        char_seq_b = []
        for sample in sample_batch:
            token_seq = np.zeros([sl], cfg.intX)
            char_seq = np.zeros([sl, self.tl], cfg.intX)

            for idx_t,(token, char_seq_v) in enumerate(zip(sample['root_node']['token_seq_digital'],
                                                           sample['root_node']['char_seq_digital'])):
                token_seq[idx_t] = token
                for idx_c, char in enumerate(char_seq_v):
                    if idx_c >= self.tl: break
                    char_seq[idx_t, idx_c] = char
            token_seq_b.append(token_seq)
            char_seq_b.append(char_seq)
        token_seq_b = np.stack(token_seq_b)
        char_seq_b = np.stack(char_seq_b)

        # tree
        op_list_b = []
        reduce_mat_b = []
        for sample in sample_batch:
            op_list = np.zeros([ol], cfg.intX)
            reduce_mat = np.zeros([ol, mc], cfg.intX)

            for idx_o, (op, reduce_list) in enumerate(zip(sample['shift_reduce_info']['op_list'],
                                                          sample['shift_reduce_info']['reduce_mat'])):
                op_list[idx_o] = op
                for idx_m, red in enumerate(reduce_list):
                    reduce_mat[idx_o, idx_m] = red
            op_list_b.append(op_list)
            reduce_mat_b.append(reduce_mat)
        op_list_b = np.stack(op_list_b)
        reduce_mat_b = np.stack(reduce_mat_b)

        # label
        sentiment_label_b = []
        for sample in sample_batch:
            sentiment_float = sample['root_node']['sentiment_label']
            sentiment_int = cfg.sentiment_float_to_int(sentiment_float)
            sentiment_label_b.append(sentiment_int)
        sentiment_label_b = np.stack(sentiment_label_b).astype(cfg.intX)

        feed_dict = {self.token_seq: token_seq_b, self.char_seq: char_seq_b,
                     self.op_list: op_list_b, self.reduce_mat: reduce_mat_b,
                     self.sentiment_label: sentiment_label_b,
                     self.is_train: True if data_type == 'train' else False}
        return feed_dict

    def step(self, sess, batch_samples, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, 'train')
        cfg.time_counter.add_start()
        if get_summary:
            loss, summary, train_op = sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)

        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        cfg.time_counter.add_stop()
        return loss, summary, train_op
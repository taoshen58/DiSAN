from configs import cfg
from src.utils.record_log import _logger
import tensorflow as tf
from src.model.template import ModelTemplate

from src.nn_utils.integration_func import multi_dimensional_attention, generate_embedding_mat,\
    directional_attention_with_dense
from src.nn_utils.nn import linear


class ModelExpEmbDirMulAttn(ModelTemplate):
    def __init__(self, token_emb_mat, glove_emb_mat, tds, cds, tl, scope):
        super(ModelExpEmbDirMulAttn, self).__init__(token_emb_mat, glove_emb_mat, tds, cds, tl, scope)
        self.update_tensor_add_ema_and_opt()

    def build_network(self):
        _logger.add()
        _logger.add('building %s neural network structure...' % cfg.network_type)
        tds, cds = self.tds, self.cds
        tl = self.tl
        tel, cel, cos, ocd, fh = self.tel, self.cel, self.cos, self.ocd, self.fh
        hn = self.hn
        bs, sl, ol, mc = self.bs, self.sl, self.ol, self.mc

        with tf.variable_scope('emb'):
            token_emb_mat = generate_embedding_mat(tds, tel, init_mat=self.token_emb_mat,
                                                   extra_mat=self.glove_emb_mat, extra_trainable=self.finetune_emb,
                                                   scope='gene_token_emb_mat')
            emb = tf.nn.embedding_lookup(token_emb_mat, self.token_seq)  # bs,sl,tel
            self.tensor_dict['emb'] = emb

        with tf.variable_scope('ct_attn'):
            rep_fw = directional_attention_with_dense(
                emb, self.token_mask, 'forward', 'dir_attn_fw',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                tensor_dict=self.tensor_dict, name='fw_attn')
            rep_bw = directional_attention_with_dense(
                emb, self.token_mask, 'backward', 'dir_attn_bw',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                tensor_dict=self.tensor_dict, name='bw_attn')

            seq_rep = tf.concat([rep_fw, rep_bw], -1)

        with tf.variable_scope('sent_enc_attn'):
            rep = multi_dimensional_attention(
                seq_rep, self.token_mask, 'multi_dimensional_attention',
                cfg.dropout, self.is_train, cfg.wd, 'relu',
                tensor_dict=self.tensor_dict, name='attn')

        with tf.variable_scope('output'):
            pre_logits = tf.nn.relu(linear([rep], hn, True, scope='pre_logits_linear',
                                          wd=cfg.wd, input_keep_prob=cfg.dropout,
                                          is_train=self.is_train))  # bs, hn
            logits = linear([pre_logits], self.output_class, False, scope='get_output',
                            wd=cfg.wd, input_keep_prob=cfg.dropout, is_train=self.is_train) # bs, 5
        _logger.done()
        return logits



from src.nn_utils.general import exp_mask_for_high_rank, mask_for_high_rank
from src.nn_utils.nn import linear, get_logits, softsel, dropout, bn_dense_layer
from src.nn_utils.rnn_cell import SwitchableDropoutWrapper
from src.nn_utils.rnn import bidirectional_dynamic_rnn
import tensorflow as tf
from src.nn_utils.general import get_last_state, add_reg_without_bias


def traditional_attention(rep_tensor, rep_mask, scope=None,
                          keep_prob=1., is_train=None, wd=0., activation='elu',
                          tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'traditional_attention'):
        rep_tensor_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                        False, wd, keep_prob, is_train)

        rep_tensor_logits = get_logits([rep_tensor_map], None, False, scope='self_attn_logits',
                                       mask=rep_mask, input_keep_prob=keep_prob, is_train=is_train)  # bs,sl
        attn_res = softsel(rep_tensor, rep_tensor_logits, rep_mask)  # bs,vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = tf.nn.softmax(rep_tensor_logits)

        return attn_res


def multi_dimensional_attention(rep_tensor, rep_mask, scope=None,
                                keep_prob=1., is_train=None, wd=0., activation='elu',
                                tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output


def directional_attention_with_dense(rep_tensor, rep_mask, direction=None, scope=None,
                                     keep_prob=1., is_train=None, wd=0., activation='elu',
                                     tensor_dict=None, name=None):
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)

    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction is None:
            direct_mask = tf.cast(tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)
        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])  # bs,sl,sl
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])  # bs,sl,sl
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)  # bs,sl,sl

        # non-linear
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map', activation,
                                 False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])  # bs,sl,sl,vec
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable('f_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            dependent = linear(rep_map_dp, ivec, False, scope='linear_dependent')  # bs,sl,vec
            dependent_etd = tf.expand_dims(dependent, 1)  # bs,1,sl,vec
            head = linear(rep_map_dp, ivec, False, scope='linear_head') # bs,sl,vec
            head_etd = tf.expand_dims(head, 2)  # bs,sl,1,vec

            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)  # bs,sl,sl,vec

            logits_masked = exp_mask_for_high_rank(logits, attn_mask)
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)

            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)  # bs,sl,vec

        with tf.variable_scope('output'):
            o_bias = tf.get_variable('o_bias',[ivec], tf.float32, tf.constant_initializer(0.))
            # input gate
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False, wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False, wd, keep_prob, is_train) +
                o_bias)
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result
            output = mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output


# -------------- rnn --------------
def contextual_bi_rnn(tensor_rep, mask_rep, hn, cell_type, only_final=False,
                      wd=0., keep_prob=1.,is_train=None, scope=None):
    """
    fusing contextual information using bi-direction rnn
    :param tensor_rep: [..., sl, vec]
    :param mask_rep: [..., sl]
    :param hn:
    :param cell_type: 'gru', 'lstm', basic_lstm' and 'basic_rnn'
    :param only_final: True or False
    :param wd:
    :param keep_prob:
    :param is_train:
    :param scope:
    :return:
    """
    with tf.variable_scope(scope or 'contextual_bi_rnn'): # correct
        reuse = None if not tf.get_variable_scope().reuse else True
        #print(reuse)
        if cell_type == 'gru':
            cell_fw = tf.contrib.rnn.GRUCell(hn, reuse=reuse)
            cell_bw = tf.contrib.rnn.GRUCell(hn, reuse=reuse)
        elif cell_type == 'lstm':
            cell_fw = tf.contrib.rnn.LSTMCell(hn, reuse=reuse)
            cell_bw = tf.contrib.rnn.LSTMCell(hn, reuse=reuse)
        elif cell_type == 'basic_lstm':
            cell_fw = tf.contrib.rnn.BasicLSTMCell(hn, reuse=reuse)
            cell_bw = tf.contrib.rnn.BasicLSTMCell(hn, reuse=reuse)
        elif cell_type == 'basic_rnn':
            cell_fw = tf.contrib.rnn.BasicRNNCell(hn, reuse=reuse)
            cell_bw = tf.contrib.rnn.BasicRNNCell(hn, reuse=reuse)
        else:
            raise AttributeError('no cell type \'%s\'' % cell_type)
        cell_dp_fw = SwitchableDropoutWrapper(cell_fw,is_train,keep_prob)
        cell_dp_bw = SwitchableDropoutWrapper(cell_bw,is_train,keep_prob)

        tensor_len = tf.reduce_sum(tf.cast(mask_rep, tf.int32), -1)  # [bs]

        (outputs_fw, output_bw), _=bidirectional_dynamic_rnn(
            cell_dp_fw, cell_dp_bw, tensor_rep, tensor_len,
            dtype=tf.float32)
        rnn_outputs = tf.concat([outputs_fw,output_bw],-1)  # [...,sl,2hn]

        if wd > 0:
            add_reg_without_bias()
        if not only_final:
            return rnn_outputs  # [....,sl, 2hn]
        else:
            return get_last_state(rnn_outputs, mask_rep)  # [...., 2hn]


# -------------- emb mat--------------
def generate_embedding_mat(dict_size, emb_len, init_mat=None, extra_mat=None,
                           extra_trainable=False, scope=None):
    """
    generate embedding matrix for looking up
    :param dict_size: indices 0 and 1 corresponding to empty and unknown token
    :param emb_len:
    :param init_mat: init mat matching for [dict_size, emb_len]
    :param extra_mat: extra tensor [extra_dict_size, emb_len]
    :param extra_trainable:
    :param scope:
    :return: if extra_mat is None, return[dict_size+extra_dict_size,emb_len], else [dict_size,emb_len]
    """
    with tf.variable_scope(scope or 'gene_emb_mat'):
        emb_mat_ept_and_unk = tf.constant(value=0, dtype=tf.float32, shape=[2, emb_len])
        if init_mat is None:
            emb_mat_other = tf.get_variable('emb_mat',[dict_size - 2, emb_len], tf.float32)
        else:
            emb_mat_other = tf.get_variable("emb_mat",[dict_size - 2, emb_len], tf.float32,
                                            initializer=tf.constant_initializer(init_mat[2:], dtype=tf.float32,
                                                                                verify_shape=True))
        emb_mat = tf.concat([emb_mat_ept_and_unk, emb_mat_other], 0)

        if extra_mat is not None:
            if extra_trainable:
                extra_mat_var = tf.get_variable("extra_emb_mat",extra_mat.shape, tf.float32,
                                                initializer=tf.constant_initializer(extra_mat,
                                                                                    dtype=tf.float32,
                                                                                    verify_shape=True))
                return tf.concat([emb_mat, extra_mat_var], 0)
            else:
                #with tf.device('/cpu:0'):
                extra_mat_con = tf.constant(extra_mat, dtype=tf.float32)
                return tf.concat([emb_mat, extra_mat_con], 0)
        else:
            return emb_mat



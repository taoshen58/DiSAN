import tensorflow as tf
from src.nn_utils.rnn import dynamic_rnn, bidirectional_dynamic_rnn
from src.nn_utils.rnn_cell import SwitchableDropoutWrapper
from src.nn_utils.general import get_last_state, add_reg_without_bias
from src.nn_utils.nn import highway_network,multi_conv1d


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


def one_direction_rnn(tensor_rep, mask_rep, hn, cell_type, only_final=False,
                      wd=0., keep_prob=1.,is_train=None, is_forward = True,scope=None):
    assert not is_forward # todo: waiting to be implemented
    with tf.variable_scope(scope or '%s_rnn' % 'forward' if is_forward else 'backward'):
        reuse = None if not tf.get_variable_scope().reuse else True
        # print(reuse)
        if cell_type == 'gru':
            cell = tf.contrib.rnn.GRUCell(hn, reuse=reuse)
        elif cell_type == 'lstm':
            cell = tf.contrib.rnn.LSTMCell(hn, reuse=reuse)
        elif cell_type == 'basic_lstm':
            cell = tf.contrib.rnn.BasicLSTMCell(hn, reuse=reuse)
        elif cell_type == 'basic_rnn':
            cell = tf.contrib.rnn.BasicRNNCell(hn, reuse=reuse)
        else:
            raise AttributeError('no cell type \'%s\'' % cell_type)
        cell_dp = SwitchableDropoutWrapper(cell, is_train, keep_prob)

        tensor_len = tf.reduce_sum(tf.cast(mask_rep, tf.int32), -1)  # [bs]

        rnn_outputs, _ = dynamic_rnn(
            cell_dp, tensor_rep, tensor_len,
            dtype=tf.float32)

        if wd > 0:
            add_reg_without_bias()
        if not only_final:
            return rnn_outputs  # [....,sl, 2hn]
        else:
            return get_last_state(rnn_outputs, mask_rep)  # [...., 2hn]


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


def token_and_char_emb(if_token_emb=True, context_token=None, tds=None, tel=None,
                       token_emb_mat=None, glove_emb_mat=None,
                       if_char_emb=True, context_char=None, cds=None, cel=None,
                       cos=None, ocd=None, fh=None, use_highway=True,highway_layer_num=None,
                       wd=0., keep_prob=1., is_train=None):
    with tf.variable_scope('token_and_char_emb'):
        if if_token_emb:
            with tf.variable_scope('token_emb'):
                token_emb_mat = generate_embedding_mat(tds, tel, init_mat=token_emb_mat,
                                                       extra_mat=glove_emb_mat,
                                                       scope='gene_token_emb_mat')

                c_token_emb = tf.nn.embedding_lookup(token_emb_mat, context_token)  # bs,sl,tel

        if if_char_emb:
            with tf.variable_scope('char_emb'):
                char_emb_mat = generate_embedding_mat(cds, cel, scope='gene_char_emb_mat')
                c_char_lu_emb = tf.nn.embedding_lookup(char_emb_mat, context_char)  # bs,sl,tl,cel

                assert sum(ocd) == cos and len(ocd) == len(fh)

                with tf.variable_scope('conv'):
                    c_char_emb = multi_conv1d(c_char_lu_emb, ocd, fh, "VALID",
                                              is_train, keep_prob, scope="xx")  # bs,sl,cocn
        if if_token_emb and if_char_emb:
            c_emb = tf.concat([c_token_emb, c_char_emb], -1)  # bs,sl,cocn+tel
        elif if_token_emb:
            c_emb = c_token_emb
        elif if_char_emb:
            c_emb = c_char_emb
        else:
            raise AttributeError('No embedding!')

    if use_highway:
        with tf.variable_scope('highway'):
            c_emb = highway_network(c_emb, highway_layer_num, True, wd=wd,
                                    input_keep_prob=keep_prob,is_train=is_train)
    return c_emb


def generate_feature_emb_for_c_and_q(feature_dict_size, feature_emb_len,
                                     feature_name , c_feature, q_feature=None, scope=None):
    with tf.variable_scope(scope or '%s_feature_emb' % feature_name):
        emb_mat = generate_embedding_mat(feature_dict_size, feature_emb_len, scope='emb_mat')
        c_feature_emb = tf.nn.embedding_lookup(emb_mat, c_feature)
        if q_feature is not None:
            q_feature_emb = tf.nn.embedding_lookup(emb_mat, q_feature)
        else:
            q_feature_emb = None
        return c_feature_emb, q_feature_emb

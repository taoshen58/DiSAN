"""
Author: Tao Shen
Email: tao.shen@student.uts.edu.au
University of Technology Sydney
"""

import tensorflow as tf
import math
N_INF = -1e12


def stacking_fast_directional_self_attention(
        rep_tensor, rep_mask, hn, head_num=8,
        is_train=None, residual_keep_prob=.8, attn_keep_prob=.8, dense_keep_prob=.9, wd=0.,  # dropout and L2
        use_direction=True, attn_self=False,
        activation_func_name='relu', dot_activation_name='exp',
        layer_num=10, scope=None
):
    """
    stacked Fast-DiSA
    :param rep_tensor: same as that in Fast-DiSA;
    :param rep_mask: same as that in Fast-DiSA;
    :param hn: same as that in Fast-DiSA;
    :param head_num: same as that in Fast-DiSA;
    :param is_train: same as that in Fast-DiSA;
    :param residual_keep_prob: float-[], dropout keep probability for residual connection;
    :param attn_keep_prob: same as that in Fast-DiSA;
    :param dense_keep_prob: same as that in Fast-DiSA;
    :param wd: same as that in Fast-DiSA;
    :param use_direction: same as that in Fast-DiSA;
    :param attn_self: same as that in Fast-DiSA;
    :param activation_func_name: same as that in Fast-DiSA;
    :param dot_activation_name: same as that in Fast-DiSA;
    :param layer_num: int-[], the number of layer stacked;
    :param scope: soc
    :return:
    """
    with tf.variable_scope(scope or 'stacking_fast_disa'):
        final_mask_ft = mask_ft_generation(rep_mask, head_num, use_direction, attn_self)
        x = rep_tensor
        for layer_idx in range(layer_num):
            with tf.variable_scope('layer_%d' % layer_idx):
                # ffn
                y = bn_dense_layer(
                    x, hn, True, 0., 'ffn', activation_func_name, False, wd, dense_keep_prob, is_train)
                x = residual_connection(x, y, is_train, residual_keep_prob, 'res_con_1')
                # self-attn
                y = fast_directional_self_attention(
                    x, rep_mask, hn, head_num, is_train, attn_keep_prob, dense_keep_prob, wd, use_direction,
                    attn_self=attn_self, use_fusion_gate=False, final_mask_ft=final_mask_ft,
                    dot_activation_name=dot_activation_name, use_input_for_attn=True, add_layer_for_multi=False,
                    activation_func_name=activation_func_name, apply_act_for_v=False, input_hn=None,
                    output_hn=hn, accelerate=True, merge_var=False, scope='fast_disa'
                )
                x = residual_connection(x, y, is_train, residual_keep_prob, 'res_con_2')

    return x


def fast_directional_self_attention(
        rep_tensor, rep_mask, hn, head_num=2,
        is_train=None, attn_keep_prob=1., dense_keep_prob=1., wd=0.,  # dropout and L2
        use_direction=True, attn_self=False, use_fusion_gate=True, final_mask_ft=None,  # direction & fusion gate
        dot_activation_name='exp', use_input_for_attn=False, add_layer_for_multi=True,
        activation_func_name='relu', apply_act_for_v=True, input_hn=None, output_hn=None,  # non-linearity
        accelerate=False, merge_var=False,
        scope=None
):
    """
    The general API for Fast Self-Attention Attention mechanism for context fusion.
    :param rep_tensor: tf.float32-[batch_size,seq_len,channels], input sequence tensor;
    :param rep_mask: tf.bool-[batch_size,seq_len], mask to indicate padding or not for "rep_tensor";
    :param hn: int32-[], hidden unit number for this attention module;
    :param head_num: int32-[]; multi-head number, if "use_direction" is set to True, this must be set to a even number,
    i.e., half for forward and remaining for backward;
    :param is_train: tf.bool-[]; This arg must be a Placehold or Tensor of Tensorflow. This may be useful if you build
    a graph for both training and testing, and you can create a Placehold to indicate training(True) or testing(False)
    and pass the Placehold into this method;
    :param attn_keep_prob: float-[], the value must be in [0.0 ,1.0] and this keep probability is for attenton dropout;
    :param dense_keep_prob: float-[], the value must be in [0.0 ,1.0] and this probability is for dense-layer dropout;
    :param wd: float-[], if you use L2-reg, set this value to be greater than 0., which will result in that the
    trainable parameters (without biases) are added to a tensorflow collection named as "reg_vars";
    :param use_direction: bool-[], for mask generation, use forward and backward direction masks or not;
    :param attn_self: bool-[], for mask generation, include attention over self or not
    :param use_fusion_gate: bool-[], use a fusion gate to dynamically combine attention results with input or not.
    :param final_mask_ft: None/tf.float-[head_num,batch_size,seq_len,seq_len], the value is whether 0 (disabled) or
    1 (enabled), set to None if you only use single layer of this method; use *mask_generation* method
    to generate one and pass it into this method if you want to stack this module for computation resources saving;
    :param dot_activation_name: str-[], "exp" or "sigmoid", the activation function name for dot product
    self-attention logits;
    :param use_input_for_attn: bool-[], if True, use *rep_tensor* to compute dot-product and s2t multi-dim self-attn
    alignment score; if False, use a tensor obtained by applying a dense layer to the *rep_tensor*, which can add the
    non-linearity for this layer;
    :param add_layer_for_multi: bool-[], if True, add a dense layer with activation func -- "activation_func_name"
    to calculate the s2t multi-dim self-attention alignment score;
    :param activation_func_name: str-[], activation function name, commonly-used: "relu", "elu", "selu";
    :param apply_act_for_v: bool-[], if or not apply the non-linearity activation function ("activation_func_name") to
    value map (same as the value map in multi-head attention);
    :param apply_act_for_v: bool-[], if apply an activation function to v in the attention;
    :param input_hn: None/int32-[], if not None, add an extra dense layer (unit num is "input_hn") with
    activation function ("activation_func_name") before attention without consideration of multi-head.
    :param output_hn: None/int32-[], if not None, add an extra dense layer (unit num is "output_hn") with
    activation function ("activation_func_name") after attention without consideration of multi-head.
    :param accelerate: bool-[], for model acceleration, we optimize and combined some matrix multiplication if using
    the accelerate (i.e., set as True), which may effect the dropout-sensitive models or tasks.
    :param merge_var: bool-[], because the batch matmul is used for parallelism of multi-head attention, if True, the
    trainable variables are declared and defined together, otherwise them are defined separately and combined together.
    :param scope: None/str-[], variable scope name.
    :return: tf.float32-[batch_size, sequence_length, out_hn], if output_hn is not None, the out_hn = "output_hn"
    otherwise out_hn = "hn"
    """
    with tf.variable_scope(scope or 'proposed_self_attention'):
        # parameters inspection
        assert hn % head_num == 0, "hn (%d) must be divisible by the number of " \
                                   "attention heads (%d)." % (hn, head_num)
        if use_direction:
            assert head_num % 2 == 0, "attention heads (%d) must be a even number when using direction." % head_num

        # input non-linearity
        if input_hn is not None:
            with tf.variable_scope("input_non_linearity"):
                rep_tensor = bn_dense_layer(
                    rep_tensor, input_hn, True, 0., 'linear_input',
                    activation_func_name, False, wd, dense_keep_prob, is_train, 1, merge_var
                )

        # position mask generate [num,bs,sl,sl]
        if final_mask_ft is None:
            final_mask_ft = mask_ft_generation(rep_mask, head_num, use_direction, attn_self)

        # dimension/channel number for each head
        head_dim = int(hn / head_num)

        if not accelerate:
            # input preparation for each head. tiling here is to make the dropout different for each head.
            rep_tensor_tl = tf.tile(tf.expand_dims(rep_tensor, 0), [head_num, 1, 1, 1])  # num,bs,sl,xx

            # calculate value map
            v = multi_head_dense_layer(  # num,bs,sl,dim
                rep_tensor_tl, head_dim, True, 0., 'v_transform',
                'linear' if not apply_act_for_v else activation_func_name,
                False, wd, dense_keep_prob, is_train, 1, merge_var
            )

            # choose the source for both dot-product self-attention and s2t multi-dim self-attention
            for_attn_score = rep_tensor_tl if use_input_for_attn else v

            q = multi_head_dense_layer(
                for_attn_score, head_dim, False, 0., 'q_transform', 'linear',
                False, wd, dense_keep_prob, is_train, 1, merge_var)
            k = multi_head_dense_layer(
                for_attn_score, head_dim, False, 0., 'k_transform', 'linear',
                False, wd, dense_keep_prob, is_train, 1, merge_var)
        else:  # use_input_for_attn, apply_act_for_v
            for_attn_score = None
            if use_input_for_attn:
                qkv_combine = bn_dense_layer(
                    rep_tensor, hn, False, 0., 'qkv_combine', 'linear',
                    False, wd, dense_keep_prob, is_train, 3, merge_var)
                q, k, v = tf.split(qkv_combine, 3, -1)
                q = split_head(q, head_num)
                k = split_head(k, head_num)
                v = split_head(v, head_num)  # num,bs,sl,dim
                if apply_act_for_v:
                    v = activation_name_to_func(activation_func_name)(v)
            else:
                v = bn_dense_layer(  # num,bs,sl,dim
                    rep_tensor, hn, True, 0., 'v_transform',
                    'linear' if not apply_act_for_v else activation_func_name,
                    False, wd, dense_keep_prob, is_train, 1, merge_var
                )
                v = split_head(v, head_num)  # num,bs,sl,dim
                if apply_act_for_v:
                    v = activation_name_to_func(activation_func_name)(v)
                qk_combine = multi_head_dense_layer(
                    v, head_dim, False, 0., 'qk_combine', 'linear',
                    False, wd, dense_keep_prob, is_train, 2, merge_var
                )
                q, k = tf.split(qk_combine, 2, -1)  # num,bs,sl,dim

        # dot-product (multi-head) self-attention
        with tf.name_scope("dot_product_attention"):
            # calculate the logits
            dot_logits = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / math.sqrt(head_dim)  # num,bs,sl,sl
            # apply activation function and positional mask to logits from the attention
            e_dot_logits = final_mask_ft * activation_name_to_func(dot_activation_name)(dot_logits)

        # s2t multi-dim self-attention
        with tf.variable_scope("s2t_multi_dim_attention"):
            if not accelerate:
                assert for_attn_score is not None
                # Add an extra dense layer with activation func
                if add_layer_for_multi:
                    tensor4multi = multi_head_dense_layer(
                        for_attn_score, head_dim, True, 0., 'tensor4multi', activation_func_name,
                        False, wd, dense_keep_prob, is_train, 1, merge_var)
                else:
                    tensor4multi = for_attn_score
                # calculate the logits
                multi_logits = multi_head_dense_layer(
                    tensor4multi, head_dim, True, 0., 'multi_logits', 'linear', False,
                    wd, dense_keep_prob, is_train, 1, merge_var)
            else:  # use_input_for_attn, add_layer_for_multi
                if use_input_for_attn:
                    tensor4multi = bn_dense_layer(
                        rep_tensor, hn, True, 0., 'tensor4multi', 'linear', False,
                        wd, dense_keep_prob, is_train, 1, merge_var
                    )
                    tensor4multi = split_head(tensor4multi, head_num)
                else:
                    tensor4multi = multi_head_dense_layer(
                        v, head_dim, True, 0., 'tensor4multi', 'linear', False,
                        wd, dense_keep_prob, is_train, 1, merge_var
                    )
                if add_layer_for_multi:
                    multi_logits = multi_head_dense_layer(
                        activation_name_to_func(activation_func_name)(tensor4multi), head_dim,
                        True, 0., 'multi_logits', 'linear', False, wd, dense_keep_prob, is_train, 1, merge_var
                    )
                else:
                    multi_logits = tf.identity(tensor4multi, name='multi_logits')

            # apply exponent to the logits
            e_multi_logits = mask_v2(tf.exp(multi_logits), rep_mask, multi_head=True, high_dim=True)  # num,bs,sl,dim

        # combine both calculated attention logists, i.e., alignment scores, and perform attention procedures
        with tf.name_scope("hybrid_attn"):
            # Z: softmax normalization term in attention probabilities calculation
            accum_z_deno = tf.matmul(e_dot_logits, e_multi_logits)  # num,bs,sl,dim
            accum_z_deno = tf.where(  # in case of NaN and Inf
                tf.greater(accum_z_deno, tf.zeros_like(accum_z_deno)),
                accum_z_deno,
                tf.ones_like(accum_z_deno)
            )
            # attention dropout
            e_dot_logits = dropout(e_dot_logits, math.sqrt(attn_keep_prob), is_train)
            e_multi_logits = dropout(e_multi_logits, math.sqrt(attn_keep_prob), is_train)
            # sum of exp(logits) \multiply attention target sequence
            rep_mul_score = v * e_multi_logits
            accum_rep_mul_score = tf.matmul(e_dot_logits, rep_mul_score)
            # calculate the final attention results
            attn_res = accum_rep_mul_score / accum_z_deno

        # using a fusion gate to dynamically combine the attention results and attention input sequence
        if use_fusion_gate:
            with tf.variable_scope('context_fusion_gate'):
                fusion_gate = multi_head_dense_layer(
                    tf.concat([v, attn_res], -1), head_dim, True, 0.,
                    'linear_fusion_gate', 'sigmoid', False, wd, dense_keep_prob, is_train, 1, merge_var
                )  # num,bs,sl,dim
                attn_res = fusion_gate * v + (1 - fusion_gate) * attn_res

        # concatenate the channels from different heads
        attn_res = combine_head(attn_res)  # bs,sl,hn

        # output non-linearity
        if output_hn is not None:
            with tf.variable_scope("output_non_linearity"):
                attn_res = bn_dense_layer(
                    attn_res, output_hn, True, 0., 'linear_output',
                    activation_func_name, False, wd, dense_keep_prob, is_train, 1, merge_var
                )

        # set un-mask sequence terms to zeros
        output = mask_v2(attn_res, rep_mask, False, True)

        return output


# ===============================================
def mask_ft_generation(rep_mask, head_num, use_direction, attn_self):
    return tf.cast(mask_generation(rep_mask, head_num, use_direction, attn_self), tf.float32)


def mask_generation(rep_mask, head_num, use_direction, attn_self):
    with tf.name_scope('mask_generation'):
        bs, sl = tf.shape(rep_mask)[0], tf.shape(rep_mask)[1]
        # regular mask
        rep_mask_epd1 = tf.expand_dims(rep_mask, 1)  # bs,1,sl
        rep_mask_epd2 = tf.expand_dims(rep_mask, 2)  # bs,sl,1
        rep_mask_mat = tf.logical_and(rep_mask_epd1, rep_mask_epd2)  # bs,sl,sl

        # position mask
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)

        if use_direction:
            comp_func = tf.greater_equal if attn_self else tf.greater
            fw_mask = comp_func(sl_row, sl_col)  # sl,sl
            bw_mask = comp_func(sl_col, sl_row)  # sl,sl
            direct_mask = tf.stack([fw_mask, bw_mask], 0)  # 2,sl,sl
            direct_mask = tf.reshape(  # num,sl,sl
                tf.tile(tf.expand_dims(direct_mask, 1), [1, int(head_num / 2), 1, 1]),  # 2,4,sl,sl
                [head_num, sl, sl])
        else:
            if not attn_self:
                direct_mask = tf.tile(tf.expand_dims(tf.not_equal(sl_row, sl_col), 0), [head_num, 1, 1])  # n,sl,sl
            else:
                raise(ValueError, "A attention overself must be avoided without fw/bw information")
        final_mask = tf.logical_and(  # num,bs,sl,sl
            tf.expand_dims(rep_mask_mat, 0),
            tf.expand_dims(direct_mask, 1))
        return final_mask


def split_head(inp_tensor, head_num, name=None):
    bs, sl = tf.shape(inp_tensor)[0], tf.shape(inp_tensor)[1]
    ivec = inp_tensor.get_shape().as_list()[-1]
    head_dim = int(ivec // head_num)
    with tf.name_scope(name or 'split_head'):
        inp_rsp = tf.reshape(inp_tensor, [bs, sl, head_num, head_dim])
        return tf.transpose(inp_rsp, [2, 0, 1, 3])  # num, bs, sl, dim


def combine_head(inp_tensor, name=None):
    head_num, head_dim = inp_tensor.get_shape().as_list()[0], inp_tensor.get_shape().as_list()[-1]
    bs, sl = tf.shape(inp_tensor)[1], tf.shape(inp_tensor)[2]
    with tf.name_scope(name or 'combine_head'):
        inp_trans = tf.transpose(inp_tensor, [1, 2, 0, 3])
        return tf.reshape(inp_trans, [bs, sl, head_num*head_dim])


def exp_mask_v2(val, m, multi_head=False, high_dim=False, name=None):
    with tf.name_scope(name or "new_exp_mask"):
        if multi_head:
            m = tf.expand_dims(m, 0)
        if high_dim:
            m = tf.expand_dims(m, -1)
        m_flt = tf.cast(m, tf.float32)
        return val + (1. - m_flt) * N_INF


def mask_v2(val, m, multi_head=False, high_dim=False, name=None):
    with tf.name_scope(name or "new_exp_mask"):
        if multi_head:
            m = tf.expand_dims(m, 0)
        if high_dim:
            m = tf.expand_dims(m, -1)
        m_flt = tf.cast(m, tf.float32)
        return val * m_flt


def multi_head_dense_layer(
        input_tensor, hn, bias, bias_start=0.0, scope=None, activation='relu',
        enable_bn=False, wd=0., keep_prob=1.0, is_train=None, dup_num=1, merge_var=False):
    assert not enable_bn
    # activation
    activation_func = activation_name_to_func(activation)

    hd_num = input_tensor.get_shape().as_list()[0]
    bs = tf.shape(input_tensor)[1]
    sl = tf.shape(input_tensor)[2]
    hd_dim = input_tensor.get_shape().as_list()[3]

    with tf.variable_scope(scope or 'multi_head_dense_layer'):
        # dropout
        input_tensor = dropout(input_tensor, keep_prob, is_train)

        if merge_var:
            weight = tf.get_variable('W', shape=[hd_num, hd_dim, hn*dup_num])
        else:
            weight_list = []
            for i in range(hd_num):
                sub_weight_list = []
                for j in range(dup_num):
                    sub_weight_list.append(tf.get_variable('W_%d_%d' % (i, j), shape=[hd_dim, hn]))
                weight_list.append(tf.concat(sub_weight_list, -1) if dup_num > 1 else sub_weight_list[0])
            weight = tf.stack(weight_list, 0)

        input_tensor_rsp = tf.reshape(input_tensor, [hd_num, bs*sl, hd_dim])  # hd_num, bs*sl, hd_dim
        out_rsp = tf.matmul(input_tensor_rsp, weight)  # hd_num, bs*sl, hn
        if bias:
            if merge_var:
                bias_val = tf.get_variable(
                        'bias', shape=[hd_num, 1, hn], dtype=tf.float32,
                        initializer=tf.constant_initializer(bias_start))
            else:
                bias_list = []
                for i in range(hd_num):
                    sub_bias_list = []
                    for j in range(dup_num):
                        sub_bias_list.append(
                            tf.get_variable(
                                'bias_%d_%d' % (i, j), shape=[1, hn], dtype=tf.float32,
                                initializer=tf.constant_initializer(bias_start)))
                    bias_list.append(tf.concat(sub_bias_list, -1) if dup_num > 1 else sub_bias_list[0])
                bias_val = tf.stack(bias_list, 0)
            out_rsp = out_rsp + bias_val   # hd_num, bs*sl, hn

        out = tf.reshape(out_rsp, [hd_num, bs, sl, hn*dup_num])

        if wd:
            tf.add_to_collection('reg_vars', weight)
        return activation_func(out)


def selu(x):
    with tf.name_scope('elu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))


def activation_name_to_func(activation_name):
    assert isinstance(activation_name, str)

    if activation_name == 'linear':
        activation_func = tf.identity
    elif activation_name == 'relu':
        activation_func = tf.nn.relu
    elif activation_name == 'elu':
        activation_func = tf.nn.elu
    elif activation_name == 'selu':
        activation_func = selu
    elif activation_name == 'sigmoid':
        activation_func = tf.nn.sigmoid
    elif activation_name == 'tanh':
        activation_func = tf.nn.tanh
    elif activation_name == 'exp':
        activation_func = tf.exp
    elif activation_name == 'log':
        activation_func = tf.log
    else:
        raise AttributeError('no activation function named as %s' % activation_name)
    return activation_func


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        if is_train is None:
            if keep_prob < 1.0:
                return tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
        else:
            if keep_prob < 1.0:
                out = tf.cond(
                    is_train,
                    lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed),
                    lambda: x
                )
                return out
        return x


# =========linear==============
def bn_dense_layer(
        input_tensor, hn, bias, bias_start=0.0, scope=None,
        activation='relu', enable_bn=False,
        wd=0., keep_prob=1.0, is_train=None, dup_num=1, merge_var=False
):
    assert len(input_tensor.get_shape().as_list()) == 3
    # activation
    activation_func = activation_name_to_func(activation)
    with tf.variable_scope(scope or 'bn_dense_layer'):
        bs, sl = tf.shape(input_tensor)[0], tf.shape(input_tensor)[1]
        input_dim = input_tensor.get_shape().as_list()[2]

        input_tensor = dropout(input_tensor, keep_prob, is_train)
        input_tensor_rsp = tf.reshape(input_tensor, [bs*sl, input_dim])

        if merge_var:
            weight = tf.get_variable('W', shape=[input_dim, hn * dup_num], dtype=tf.float32)
        else:
            weight_list = []
            for i in range(dup_num):
                weight_list.append(tf.get_variable('W_%d' % i, shape=[input_dim, hn]))
            weight = tf.concat(weight_list, -1)
        output_rsp = tf.matmul(input_tensor_rsp, weight)

        if bias:
            if merge_var or dup_num == 1:
                bias_val = tf.get_variable(
                    'bias', shape=[hn * dup_num], dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start)
                )
            else:
                bias_list = []
                for i in range(dup_num):
                    bias_list.append(
                        tf.get_variable(
                            'bias_%d' % i, shape=[hn], dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_start))
                    )
                bias_val = tf.concat(bias_list, -1)
            output_rsp += bias_val

        output = tf.reshape(output_rsp, [bs, sl, hn*dup_num])

        if enable_bn:
            output = tf.contrib.layers.batch_norm(
                output, center=True, scale=True, is_training=is_train,
                updates_collections=None,  decay=0.9,
                scope='bn')

        if wd:
            tf.add_to_collection('reg_vars', weight)

        return activation_func(output)


def bn_dense_layer_conv(
        input_tensor, hn, bias, bias_start=0.0, scope=None,
        activation='relu', enable_bn=False,
        wd=0., keep_prob=1.0, is_train=None, dup_num=1, merge_var=False
):
    assert len(input_tensor.get_shape().as_list()) == 3
    # activation
    activation_func = activation_name_to_func(activation)
    with tf.variable_scope(scope or 'bn_dense_layer'):
        input_dim = input_tensor.get_shape().as_list()[2]

        # dropout
        input_tensor = dropout(input_tensor, keep_prob, is_train)

        if merge_var:
            weight = tf.get_variable('W', shape=[input_dim, hn * dup_num], dtype=tf.float32)
        else:
            weight_list = []
            for i in range(dup_num):
                weight_list.append(tf.get_variable('W_%d' % i, shape=[input_dim, hn]))
            weight = tf.concat(weight_list, -1)

        matrix = tf.expand_dims(tf.expand_dims(weight, 0), 1)
        input_tensor_epd = tf.expand_dims(input_tensor, 1)
        output_epd = tf.nn.convolution(
            input_tensor_epd, matrix, "VALID", data_format="NHWC")
        output = tf.squeeze(output_epd, [1])

        if bias:
            if merge_var or dup_num == 1:
                bias_val = tf.get_variable(
                    'bias', shape=[hn * dup_num], dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start)
                )
            else:
                bias_list = []
                for i in range(dup_num):
                    bias_list.append(
                        tf.get_variable(
                            'bias_%d' % i, shape=[hn], dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_start))
                    )
                bias_val = tf.concat(bias_list, -1)
            output += bias_val

        if enable_bn:
            output = tf.contrib.layers.batch_norm(
                output, center=True, scale=True, is_training=is_train,
                updates_collections=None, decay=0.9,
                scope='bn')

        if wd:
            tf.add_to_collection('reg_vars', weight)

        return activation_func(output)


# ------------------ multi-dim Source2token ------------
def multi_dim_souce2token_self_attn(rep_tensor, rep_mask, scope=None,
                                    keep_prob=1., is_train=None, wd=0., activation='elu',
                                    tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train, 1, False)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train, 1, False)
        map2_masked = exp_mask_v2(map2, rep_mask, high_dim=True)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output


# --------------- residual connection -------------
def residual_connection(x, y, is_train=None, residual_keep_prob=1., scope=None):
    with tf.variable_scope(scope or 'residual_connection'):
        y = dropout(y, residual_keep_prob, is_train)
        return layer_norm(x + y, scope='layer_norm')


def layer_norm(inputs, epsilon=1e-6, scope=None):
    with tf.variable_scope(scope or "layer_norm"):
        channel_size = inputs.get_shape().as_list()[-1]

        scale = tf.get_variable("scale", shape=[channel_size],
                                initializer=tf.ones_initializer())

        offset = tf.get_variable("offset", shape=[channel_size],
                                 initializer=tf.zeros_initializer())

        mean = tf.reduce_mean(inputs, axis=-1, keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1,
                                  keep_dims=True)

        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)

        return norm_inputs * scale + offset
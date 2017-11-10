import tensorflow as tf
from src.nn_utils.nn import softmax, softsel, linear, get_logits, linear_3d, softsel_with_dropout
from src.nn_utils.general import exp_mask, mask_for_high_rank
from src.nn_utils.general import mask as normal_mask
import math


def normal_attention(tensor_base, tensor_to_attend,
                     mask_for_tensor_base,
                     mask_for_tensor_to_attend,
                     similarity_method='inner', hn=100,
                     use_pooling=False, pooling_method='max',
                     reverse=False, scope=None):
    """
    normal_attention for attention strategy 2 
    :param tensor_base: rank 3 [bs,sl,vec]
    :param tensor_to_attend: rank 3 [bs,ql,vec]
    :param mask_for_tensor_base: [bs,ql]
    :param mask_for_tensor_to_attend: [bs,sl]
    :param similarity_method: 'inner' 'tri_linear' 'map_linear'
    :param hn: some method need 
    :param use_pooling: True or False
    :param pooling_method: 'max' or 'mean'
    :param reverse: if use strategy 3
    :param scope: 
    :return: use_pooling==True: [bs,sl,hn] else [bs,hn]
    """
    with tf.variable_scope(scope or 'normal_attention'):
        # --------parameters--------
        t_main = tensor_base  # [bs,sl,vec]
        t_sec = tensor_to_attend  # [bs,ql,vec]
        mask_main = mask_for_tensor_base  # [bs,sl]
        mask_sec = mask_for_tensor_to_attend  # [bs,ql]

        bs, sl, vec = tf.shape(t_main)[0], tf.shape(t_main)[1], tf.shape(t_main)[2]
        ql = tf.shape(t_sec)[1]
        # -------------------------------
        # --------similarity_mat--------
        mask_main_etd = tf.expand_dims(mask_main, 2)  # bs,sl,1
        mask_sec_etd = tf.expand_dims(mask_sec, 1)  # bs,1,ql
        mask_similarity_mat = tf.logical_and(mask_main_etd, mask_sec_etd)  # bs,sl,ql
        if similarity_method == 'inner':
            t_main_etd = tf.expand_dims(t_main, 2)  # bs,sl,1,vec
            t_sec_etd = tf.expand_dims(t_sec, 1)  # bs,1,ql,vec
            similarity_mat = tf.reduce_sum(t_main_etd*t_sec_etd, -1)  # bs,sl,ql
        elif similarity_method == 'tri_linear':
            t_main_tiled = tf.tile(tf.expand_dims(t_main, 2), [1, 1, ql, 1])  # bs,sl,ql,vec
            t_sec_tiled = tf.tile(tf.expand_dims(t_sec, 1), [1, sl, 1, 1])  # bs,sl,ql,vec
            similarity_mat = get_logits([t_main_tiled, t_sec_tiled], None, False,
                                        scope='tri_linear_tri_linear', func='tri_linear')
        elif similarity_method == 'map_linear':
            t_main_map = tf.nn.relu(linear([t_main], hn, True, scope='linear_map_main'))
            t_sec_map = tf.nn.relu(linear([t_sec], hn, True, scope='linear_map_sec'))
            t_main_map_etd = tf.expand_dims(t_main_map, 2)  # bs,sl,1,hn
            t_sec_map_etd = tf.expand_dims(t_sec_map, 1)  # bs,1,ql,hn
            similarity_mat = tf.reduce_sum(t_main_map_etd * t_sec_map_etd, -1)  # bs,sl,ql
        else:
            raise AttributeError('No similarity matrix calculation method \'%s\'' % similarity_method)
        # -------------------------------
        if use_pooling:
            # pool mat along -2
            if pooling_method == 'max':
                pooling_out = tf.reduce_max(exp_mask(similarity_mat, mask_similarity_mat), -2)  # bs,sl,ql -> bs,ql
            elif pooling_method == 'mean':
                sum_out = tf.reduce_sum(normal_mask(similarity_mat, mask_similarity_mat), -2)  # bs,sl,ql -> bs,ql
                num = tf.reduce_sum(tf.cast(mask_similarity_mat, tf.int32), -2)  # bs,ql
                num = tf.where(tf.equal(num, tf.zeros_like(num, tf.int32)),
                               tf.ones_like(num, tf.int32), num)
                pooling_out = sum_out / tf.cast(num, tf.float32)  # bs,ql
            else:
                raise AttributeError('No pooling method \'%s\'' % pooling_method)
            return softsel(t_sec, pooling_out, mask_sec)  # bs,ql,vec -> bs,ql
        else:
            t_sec_tiled = tf.tile(tf.expand_dims(t_sec, 1), [1, sl, 1, 1])  # bs,sl,ql,vec
            # target: q_tiled:[bs,sl,ql,hn]; logits: [bs,sl,ql]
            if not reverse:
                out = normal_softsel(t_sec_tiled, similarity_mat, mask_similarity_mat)
            else:
                out = reverse_softsel(t_sec_tiled, similarity_mat, mask_similarity_mat)
            return out  # bs,sl,vec


def self_align_attention(rep_tensor, mask, scope=None, simplify=True, hn=None):  # correct
    """
    attention strategy 4: self * self => attention self
    :param rep_tensor: rank is three [bs,sl,hn]
    :param mask: [bs,sl] tf.bool
    :param scope
    :param simplify:
    :return:  attended tensor [bs,sl,hn]
    """
    with tf.name_scope(scope or 'self_attention'):
        bs = tf.shape(rep_tensor)[0]
        sl = tf.shape(rep_tensor)[1]
        #vec = tf.shape(rep_tensor)[2]
        ivec = rep_tensor.get_shape().as_list()[-1]

        to_be_attended = tf.tile(tf.expand_dims(rep_tensor, 1), [1, sl, 1, 1])
        if not simplify:
            assert hn is not None
            rep_tensor = tf.nn.relu(linear([rep_tensor], hn, True, 0., 'linear_transform'))
        # 1. self alignment
        mask_tiled_sec = tf.tile(tf.expand_dims(mask, 1), [1, sl, 1])  # bs,sl,sl
        mask_tiled_mian = tf.tile(tf.expand_dims(mask, 2), [1, 1, sl])  # bs,sl,sl
        mask_tiled = tf.logical_and(mask_tiled_sec, mask_tiled_mian)
        input_sec = tf.tile(tf.expand_dims(rep_tensor, 1), [1, sl, 1, 1])  # bs,1-sl,sl,hn
        input_main = tf.tile(tf.expand_dims(rep_tensor, 2), [1, 1, sl, 1])  # bs,sl,1-sl,hn
        # self_alignment = tf.reduce_sum(input_sec * input_main, -1)  # bs,sl,sl
        self_alignment = (1.0 / ivec) * tf.reduce_sum(input_sec * input_main, -1)  # bs,sl,sl
        # 2. generate diag~/ mat
        # diag = tf.expand_dims(
        #     tf.cast(tf.logical_not(
        #         tf.cast(
        #             tf.diag(
        #                 tf.ones([sl], tf.int32)), tf.bool)
        #     ), tf.float32), 0)  # 1,sl,sl
        diag = tf.expand_dims(tf.logical_not(
                tf.cast(tf.diag(tf.ones([sl], tf.int32)), tf.bool)), 0)  # 1,sl,sl
        diag = tf.tile(diag, [bs, 1, 1])  # bs, sl, sl
        # self_alignment = self_alignment * diag  # bs,sl,sl
        # 3. attend data
        context = softsel(to_be_attended, self_alignment, tf.logical_and(mask_tiled, diag))  # [bs,sl,sl],  bs,sl,hn
        return context


def self_choose_attention(rep_tensor, rep_mask, hn,  # correct
                          keep_prob=1., is_train=None, scope=None, simplify=False):
    """
    self soft choose attention with 
    :param rep_tensor: rank must be 3 [bs,sl,hn]
    :param rep_mask: [bs,sl]
    :param hn: 
    :param keep_prob: 
    :param is_train: 
    :param scope:
    :param simplify
    :return: 
    """
    with tf.variable_scope(scope or 'self_choose_attention'):
        if not simplify:
            rep_tensor_map = tf.nn.relu(linear([rep_tensor], hn, True, scope='linear_map',
                                        input_keep_prob=keep_prob, is_train=is_train))
        else:
            rep_tensor_map = tf.identity(rep_tensor)
        rep_tensor_logits = get_logits([rep_tensor_map], None, False, scope='self_attn_logits',
                                       mask=rep_mask, input_keep_prob=keep_prob, is_train=is_train)  # bs,sl
        attn_res = softsel(rep_tensor, rep_tensor_logits, rep_mask)  # bs,vec
        return attn_res


# ### -------------------------- ###
# Separated Attention 1. similarity mat  2. attention : for space and time efficiency


def gene_similarity_mat_and_mask(tensor_row, tensor_col,
                                 mask_for_tensor_row,
                                 mask_for_tensor_col,
                                 similarity_method='inner', hn=100, scope = None):
    with tf.variable_scope(scope or 'gene_similarity_mat_and_mask'):
        # --------parameters--------
        t_main = tensor_row  # [bs,sl,vec]
        t_sec = tensor_col  # [bs,ql,vec]
        mask_main = mask_for_tensor_row  # [bs,sl]
        mask_sec = mask_for_tensor_col  # [bs,ql]

        bs, sl, vec = tf.shape(t_main)[0], tf.shape(t_main)[1], tf.shape(t_main)[2]
        ql = tf.shape(t_sec)[1]
        # -------------------------------
        # --------similarity_mat--------
        mask_main_etd = tf.expand_dims(mask_main, 2)  # bs,sl,1
        mask_sec_etd = tf.expand_dims(mask_sec, 1)  # bs,1,ql
        mask_similarity_mat = tf.logical_and(mask_main_etd, mask_sec_etd)  # bs,sl,ql
        if similarity_method == 'inner':
            t_main_etd = tf.expand_dims(t_main, 2)  # bs,sl,1,vec
            t_sec_etd = tf.expand_dims(t_sec, 1)  # bs,1,ql,vec
            similarity_mat = tf.reduce_sum(t_main_etd*t_sec_etd, -1)  # bs,sl,ql
        elif similarity_method == 'tri_linear':
            t_main_tiled = tf.tile(tf.expand_dims(t_main, 2), [1, 1, ql, 1])  # bs,sl,ql,vec
            t_sec_tiled = tf.tile(tf.expand_dims(t_sec, 1), [1, sl, 1, 1])  # bs,sl,ql,vec
            similarity_mat = get_logits([t_main_tiled, t_sec_tiled], None, False,
                                        scope='tri_linear_tri_linear', func='tri_linear')
        elif similarity_method == 'map_linear':
            t_main_map = tf.nn.relu(linear([t_main], hn, True, scope='linear_map_main'))
            t_sec_map = tf.nn.relu(linear([t_sec], hn, True, scope='linear_map_sec'))
            t_main_map_etd = tf.expand_dims(t_main_map, 2)  # bs,sl,1,hn
            t_sec_map_etd = tf.expand_dims(t_sec_map, 1)  # bs,1,ql,hn
            similarity_mat = tf.reduce_sum(t_main_map_etd * t_sec_map_etd, -1)  # bs,sl,ql
        else:
            raise AttributeError('No similarity matrix calculation method \'%s\'' % similarity_method)

        return similarity_mat, mask_similarity_mat


def attention_with_similarity_mat(similarity_mat, mask_similarity_mat,
                                  tensor_to_attend, mask_for_tensor_to_attend,
                                  use_pooling=False, pooling_method='max',
                                  reverse=False, scope=None):
    if use_pooling:
        # pool mat along -2
        pooling_out = pooling_with_mask(similarity_mat, mask_similarity_mat, -2, pooling_method)  # bs,ql
        return softsel(tensor_to_attend, pooling_out, mask_for_tensor_to_attend)  # bs,ql,vec -> bs,vec
    else:
        t_sec_tiled = tf.tile(tf.expand_dims(tensor_to_attend, 1),
                              [1, tf.shape(similarity_mat)[-2], 1, 1])  # bs,sl,ql,vec
        # target: q_tiled:[bs,sl,ql,hn]; logits: [bs,sl,ql]
        if not reverse:
            out = normal_softsel(t_sec_tiled, similarity_mat, mask_similarity_mat)
        else:
            out = reverse_softsel(t_sec_tiled, similarity_mat, mask_similarity_mat)
        return out  # bs,sl,vec


def multi_head_attention(key, value, query,
                         hn, split_num,
                         scope=None):
    with tf.variable_scope(scope or 'multi_head_attention'):
        pass


def multi_self_choose_attention(tensor_rep, tensor_mask, hn, channel_num,
                                wd, keep_prob, is_train, scope=None):
    bs, sl, vec = tf.shape(tensor_rep)[0], tf.shape(tensor_rep)[1], tf.shape(tensor_rep)[2]
    fixed_shape = tensor_rep.get_shape().as_list()
    ibs = fixed_shape[0] or bs
    isl = fixed_shape[1] or sl
    ivec = fixed_shape[2]
    each_hn = int(hn / channel_num)
    with tf.variable_scope(scope or 'multi_self_choose_attention'):
        tensor_rep_re = tf.reshape(tensor_rep, [ibs*isl, ivec])
        tensor_rep_re = tf.tile(tf.expand_dims(tensor_rep_re, 0), [channel_num, 1, 1])  # n,bs*sl,vec
        tensor_map_re = linear_3d(tensor_rep_re, each_hn, True, 0., 'linear_3d_map',
                                  False, wd, keep_prob, is_train)  # n,bs*sl,ehn
        tensor_map = tf.reshape(tensor_map_re, [channel_num, ibs, isl, each_hn])  # n,bs, sl,ehn
        mask_tile = tf.tile(tf.expand_dims(tensor_mask, 0), [channel_num, 1, 1])

        # attention score
        attn_pre = linear_3d(tensor_map_re, each_hn, True, 0., 'linear_3d_pre', False,
                             wd, keep_prob, is_train)  # n,bs*sl,ehn
        attn_score = linear_3d(attn_pre, 1, True, 0., 'linear_3d_logits', True,
                               wd, keep_prob, is_train)  # n,bs*sl
        attn_score = tf.reshape(attn_score, [channel_num, ibs, isl])

        # execute attention
        if False:
            output = softsel_with_dropout(tensor_map, attn_score, mask_tile, keep_prob, is_train)  # n,bs,ehn
        else:
            output = softsel(tensor_map, attn_score, mask_tile)
        output = tf.transpose(output, [1, 0, 2])  # bs, n, ehn
        return tf.reshape(output, [ibs, channel_num*each_hn])















# ### -------------------------- ###


def normal_softsel(target, logits, mask=None, scope=None):
    """
    :param target: [ ..., J, d] dtype=float #(b,sn,sl,ql,d)
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "normal_softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out


def reverse_softsel(target, logits, mask=None, scope=None):

    with tf.name_scope(scope or "reverse_softsel"):
        logits_rank = len(logits.get_shape().as_list())
        if mask is not None:
            logits = exp_mask(logits, mask)
        a = tf.nn.softmax(logits, logits_rank-2)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out


def pooling_with_mask(rep_tensor, rep_mask, dim = -1, pooling_method='max', scope=None):
    # rep_tensor and rep_mask must have sampe shape
    with tf.name_scope(scope or '%s_pooling_with_mask_'%pooling_method):
        if pooling_method == 'max':
            pooling_out = tf.reduce_max(exp_mask(rep_tensor, rep_mask), dim)  # bs,sl,ql -> bs,xl
        elif pooling_method == 'mean':
            sum_out = tf.reduce_sum(normal_mask(rep_tensor, rep_mask), dim)  # bs,sl,ql -> bs,xl
            num = tf.reduce_sum(tf.cast(rep_tensor, tf.int32), dim)  # bs,xl
            num = tf.where(tf.equal(num, tf.zeros_like(num, tf.int32)),
                           tf.ones_like(num, tf.int32), num)
            pooling_out = sum_out / tf.cast(num, tf.float32)  # bs,xl
        else:
            raise AttributeError('No pooling method \'%s\'' % pooling_method)
        return pooling_out



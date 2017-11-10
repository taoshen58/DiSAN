import tensorflow as tf
from functools import reduce
from operator import mul

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def get_last_state(rnn_out_put, mask): # correct
    '''
    get_last_state of rnn output
    :param rnn_out_put: [d1,d2,dn-1,max_len,d]
    :param mask: [d1,d2,dn-1,max_len]
    :return: [d1,d2,dn-1,d]
    '''
    rnn_out_put_flatten = flatten(rnn_out_put, 2)# [X, ml, d]
    mask_flatten = flatten(mask,1) # [X,ml]
    idxs = tf.reduce_sum(tf.cast(mask_flatten,tf.int32),-1) - 1 # [X]
    indices = tf.stack([tf.range(tf.shape(idxs)[0]), idxs], axis=-1) #[X] => [X,2]
    flatten_res = tf.expand_dims(tf.gather_nd(rnn_out_put_flatten, indices),-2 )# #[x,d]->[x,1,d]
    return tf.squeeze(reconstruct(flatten_res,rnn_out_put,2),-2) #[d1,d2,dn-1,1,d] ->[d1,d2,dn-1,d]


def expand_tile(tensor,pattern,tile_num = None, scope=None): # todo: add more func
    with tf.name_scope(scope or 'expand_tile'):
        assert isinstance(pattern,(tuple,list))
        assert isinstance(tile_num,(tuple,list)) or tile_num is None
        assert len(pattern) == len(tile_num) or tile_num is None
        idx_pattern = list([(dim, p) for dim, p in enumerate(pattern)])
        for dim,p in idx_pattern:
            if p == 'x':
                tensor = tf.expand_dims(tensor,dim)
    return tf.tile(tensor,tile_num) if tile_num is not None else tensor


def get_initializer(matrix):
    def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
    return _initializer


def mask(val, mask, name=None):
    if name is None:
        name = 'mask'
    return tf.multiply(val, tf.cast(mask, 'float'), name=name)


def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32), name=name or 'mask_for_high_rank')


def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)


def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val, (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # original shape
    tensor_shape = tensor.get_shape().as_list() # current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))] #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def add_wd(wd, scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    with tf.name_scope("weight_decay"):
        for var in variables:
            counter+=1
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                       name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
            tf.add_to_collection('losses', weight_decay)
        return counter


def add_wd_without_bias(wd, scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    with tf.name_scope("weight_decay"):
        for var in variables:
            if len(var.get_shape().as_list()) <= 1: continue
            counter += 1
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                       name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
            tf.add_to_collection('losses', weight_decay)
        return counter


def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter


def add_var_reg(var):
    tf.add_to_collection('reg_vars', var)


def add_wd_for_var(var, wd):
    with tf.name_scope("weight_decay"):
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                   name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
        tf.add_to_collection('losses', weight_decay)


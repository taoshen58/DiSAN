# Fast Directional Self-Attention (Fast-DiSA) Mechanism

This is the Tensorflow implementation for **Fast-DiSA** and **Stacking Fast-DiSA** that is a time-efficient and memory-friendly multi-dim token2token self-attention mechanism for context fusion. Fast-DiSA can be regarded as a alternative to RNNs and multi-head self-attention.

The details of this model are elaborated in [this paper](https://arxiv.org/abs/1805.00912).

# How to Use
just download [*fast_disa.py*](https://github.com/taoshen58/DiSAN/tree/master/Fast-DiSA/fast_disa.py) and add the line below to your model script file:

    from fast_disa import fast_directional_self_attention, mask_ft_generation, stacking_fast_directional_self_attention
    
    
Then, follow the API below. 


## API

### For `fast_directional_self_attention`

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
    
### For `stacking_fast_directional_self_attention`
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
    :return: tf.float32-[batch_size, sequence_length, hn]



## Hyper-Parameters Suggestion 

* param `accelerate` should be set to `True` if the model or task is dropout-sensitive.
* param `use_direction` should be set to `False` when the input is not order-related.
* The hyper-params choosing for both single layer and stacked layer is detailed in the Table below. The reason why we choose params like this is that we need a fastest and simplest fast-disa model for the stacking. Note that the suggested hyper-params below is only for `fast_directional_self_attention`, and the suggested hyper-params have been applied in `stacking_fast_directional_self_attention`.

| Hyper-Params | For Single Layer | For Stacked Layer |
| --- | --- | --- |
| use_direction | True | True |
| attn_self | False | False (Depends on task) |
| use_fusion_gate | True | False |
| final_mask_ft | None | invoke `mask_ft_generation` |
| dot_activation_name | 'sigmoid' | 'exp' |
| use_input_for_attn | False | True |
| add_layer_for_multi | True | False |
| activation_func_name | as you will | as you will |
| apply_act_for_v | True | False |
| input_hn | None | None |
| output_hn | None | Set to `hn` for residual connection |
| accelerate | False | True |
| merge_var | False | True |

## TODO List
1. ~~release the single layer version of Fast-DiSA~~
2. ~~release the stacking version of Fast-DiSA (like the deep model in the Transformer)~~
3. ~~Projects codes for some NLP tasks~~

## Contact Information
Email: [tao.shen@student.uts.edu.au](mailto:tao.shen@student.uts.edu.au)
Feel free to contact me or open an issue if you have any question or encounter any bug!





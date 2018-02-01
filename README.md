# Directional Self-Attention Network
* This repo is the codes of [DiSAN: Directional Self-Attention Network for RNN/CNN-free Language Understanding](https://arxiv.org/abs/1709.04696).
* This is python based codes implementation under tensorflow 1.2 DL framework.
* The leaderboard of Stanford Natural Language Inference is available [here](https://nlp.stanford.edu/projects/snli/).
* Please contact [Tao Shen](Tao.Shen@student.uts.edu.au) or open an issue for questions/suggestions.


**Cite this paper using BibTex:**

    @inproceedings{shen2018disan,
    Author = {Shen, Tao and Zhou, Tianyi and Long, Guodong and Jiang, Jing and Pan, Shirui and Zhang, Chengqi},
    Booktitle = {AAAI Conference on Artificial Intelligence},
    Title = {DISAN: Directional self-attention network for rnn/cnn-free language understanding},
    Year = {2018}
    }


## Overall Requirements
* Python3 (verified on 3.5.2, or Anaconda3 4.2.0) 
* tensorflow>=1.2

#### Python Packages:

* numpy

-------
### This repo includes three part as follows:
1. Directionnal Self-Attention Network independent file -> file disan.py
2. DiSAN implementation for Stanford Natural Language Inference -> dir SNLI_disan
3. DiSAN implementation for Stanford Sentiment Classification -> dir SST_disan

__The Usage of *disan.py* will be introduced below, and as for the implementation of SNLI and SST, please enter corresponding folder for further introduction.__

__And, Code for the other experiments (e.g. SICK, MPQA, CR etc.) appeared in the paper is under preparation.__

-------
## Usage of disan.py

### Parameters:

* param **rep\_tensor**: 3D tensorflow dense float tensor [batch\_size, max\_len, dim]
* param **rep\_mask**: 2D tensorflow bool tensor as mask for rep\_tensor, [batch\_size, max\_len]
* param **scope**: tensorflow variable scope
* param **keep\_prob**: float, dropout keep probability
* param **is\_train**: tensorflow bool scalar
* param **wd**: if wd>0, add related tensor to tf collectoion "weight_decay" for further l2 decay
* param **activation**: disan activation function [elu|relu|selu]
* param **tensor\_dict**: a dict to record disan internal attention result (insignificance)
* param **name**: record name suffix (insignificance)

### Output:
2D tensorflow dense float tensor, which shape is [batch\_size, dim] as the encoding result for each sentence.

------
## Acknowledgements
* Some basic neural networks are copied from [Minjoon's Repo](https://github.com/allenai/bi-att-flow), including RNN cell, dropout-able dynamic RNN etc.


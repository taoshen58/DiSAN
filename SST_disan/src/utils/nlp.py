import nltk
import os
import numpy as np
import string
import re
from collections import Counter
import nltk

# --------------------2d spans -------------------
# read : span for each token -> char level
def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            if text.find(token, cur_idx) < 0:
                print(tokens)
                print("{} {} {}".format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss


# read
def get_word_span(context, wordss, start, stop):
    spanss = get_2d_spans(context, wordss)  # [[(start,end),...],...] -> char level
    idxs = []
    for sent_idx, spans in enumerate(spanss):
        for word_idx, span in enumerate(spans):
            if not (stop <= span[0] or start >= span[1]):
                idxs.append((sent_idx, word_idx))

    assert len(idxs) > 0, "{} {} {} {}".format(context, spanss, start, stop)
    return idxs[0], (idxs[-1][0], idxs[-1][1] + 1)  # (sent_start, token_start) --> (sent_stop, token_stop+1)


def get_word_idx(context, wordss, idx):
    spanss = get_2d_spans(context, wordss)  # [[(start,end),...],...] -> char level
    return spanss[idx[0]][idx[1]][0]

# ----------------- 1d span-----------------------

def get_1d_spans(text, token_seq):
    spans = []
    curIdx = 0
    for token in token_seq:
        token = token.replace('\xa0',' ')
        findRes = text.find(token,curIdx)
        if findRes < 0:
            raise RuntimeError('{} {} {}'.format(token,curIdx,text))
        curIdx = findRes
        spans.append((curIdx, curIdx+len(token)))
        curIdx += len(token)
    return spans


def get_word_idxs_1d(context, token_seq, char_start_idx, char_end_idx):
    """
    0 based 
    :param context: 
    :param token_seq: 
    :param char_start_idx: 
    :param char_end_idx: 
    :return: 0-based token index sequence in the tokenized context.
    """
    spans = get_1d_spans(context,token_seq)
    idxs = []
    for wordIdx, span in enumerate(spans):
        if not (char_end_idx <= span[0] or char_start_idx >= span[1]):
            idxs.append(wordIdx)
    assert len(idxs) > 0, "{} {} {} {}".format(context, token_seq, char_start_idx, char_end_idx)
    return idxs


def get_start_and_end_char_idx_for_word_idx_1d(context, token_seq, word_idx_seq):
    '''
    0 based 
    :param context: 
    :param token_seq: 
    :param word_idx_seq: 
    :return: 
    '''
    spans = get_1d_spans(context, token_seq)
    correct_spans = [span for idx,span in enumerate(spans) if idx in word_idx_seq]

    return correct_spans[0][0],correct_spans[-1][-1]


# ----------------- for node target idx -----------------------
def calculate_idx_seq_f1_score(input_idx_seq, label_idx_seq, recall_factor=1.):
    assert len(input_idx_seq) > 0 and len(label_idx_seq)>0
    # recall
    recall_counter = sum(1 for label_idx in label_idx_seq if label_idx in input_idx_seq)
    precision_counter = sum(1 for input_idx in input_idx_seq if input_idx in label_idx_seq)

    recall = 1.0*recall_counter/ len(label_idx_seq)
    precision = 1.0*precision_counter / len(input_idx_seq)

    recall = recall/recall_factor

    if recall + precision <= 0.:
        return 0.
    else:
        return 2.*recall*precision / (recall + precision)


def get_best_node_idx(node_and_leaf_pair, answer_token_idx_seq, recall_factor=1.):
    """
    all index in this function is 1 bases
    :param node_and_leaves_pair: 
    :param answer_token_idx_seq: 
    :return: 
    """
    f1_scores = []
    for node_idx, leaf_idx_seq in node_and_leaf_pair:
        f1_scores.append(calculate_idx_seq_f1_score(leaf_idx_seq,answer_token_idx_seq,
                                                    recall_factor))
    max_idx = np.argmax(f1_scores)
    return node_and_leaf_pair[max_idx][0]

# ------------------ calculate text f1-------------------

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def tokenize(text):
        return ' '.join(nltk.word_tokenize(text))

    return white_space_fix(remove_articles(remove_punc(lower(tokenize(s)))))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def check_rebuild_quality(prediction,ground_truth):
    em = exact_match_score(prediction,ground_truth)
    f1 = f1_score(prediction, ground_truth)
    return em,f1


def dynamic_length(lengthList, ratio, add=None, security = True, fileName=None):
    ratio = float(ratio)
    if add is not None:
        ratio += add
        ratio = ratio if ratio < 1 else 1
    if security:
        ratio = ratio if ratio < 0.99 else 0.99
    def calculate_dynamic_len(pdf ,ratio_ = ratio):
        cdf = []
        previous = 0
        # accumulate
        for len ,freq in pdf:
            previous += freq
            cdf.append((len, previous))
        # calculate
        for len ,accu in cdf:
            if 1.0 * accu/ previous >= ratio_:  # satisfy the condition
                return len, cdf[-1][0]
        # max
        return cdf[-1][0], cdf[-1][0]

    pdf = dict(nltk.FreqDist(lengthList))
    pdf = sorted(pdf.items(), key=lambda d: d[0])

    if fileName is not None:
        with open(fileName, 'w') as f:
            for len, freq in pdf:
                f.write('%d\t%d' % (len, freq))
                f.write(os.linesep)

    return calculate_dynamic_len(pdf, ratio)


def dynamic_keep(collect,ratio,fileName=None):

    pdf = dict(nltk.FreqDist(collect))
    pdf = sorted(pdf.items(), key=lambda d: d[1],reverse=True)

    cdf = []
    previous = 0
    # accumulate
    for token, freq in pdf:
        previous += freq
        cdf.append((token, previous))
        # calculate
    for idx, (token, accu) in enumerate(cdf):
        keepAnchor = idx
        if 1.0 * accu / previous >= ratio:  # satisfy the condition
            break

    tokenList=[]
    for idx, (token, freq) in enumerate(pdf):
        if idx > keepAnchor: break
        tokenList.append(token)


    if fileName is not None:
        with open(fileName, 'w') as f:
            for idx, (token, freq) in enumerate(pdf):
                f.write('%d\t%d' % (token, freq))
                f.write(os.linesep)

                if idx == keepAnchor:
                    print(os.linesep*20)

    return tokenList


def gene_question_explicit_class_tag(question_token):
    classes = ['what', 'how', 'who', 'when', 'which', 'where', 'why', 'whom', 'whose',
               ['am', 'is', 'are', 'was', 'were']]
    question_token = [token.lower() for token in question_token]

    for idx_c, cls in enumerate(classes):
        if not isinstance(cls, list):
            if cls in question_token:
                return idx_c
        else:
            for ccls in cls:
                if ccls in question_token:
                    return idx_c
    return len(classes)


def gene_token_freq_info(context_token, question_token):
    def look_up_dict(t_dict, t):
        try:
            return t_dict[t]
        except KeyError:
            return 0
    context_token_dict = dict(nltk.FreqDist(context_token))
    question_token_dict = dict(nltk.FreqDist(question_token))

    # context tokens in context and question dicts
    context_tf = []
    for token in context_token:
        context_tf.append((look_up_dict(context_token_dict, token), look_up_dict(question_token_dict, token)))

    # question tokens in context and question dicts
    question_tf = []
    for token in context_token:
        question_tf.append((look_up_dict(context_token_dict, token), look_up_dict(question_token_dict, token)))

    return {'context':context_tf, 'question':question_tf}



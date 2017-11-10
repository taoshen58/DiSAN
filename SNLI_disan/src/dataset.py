from configs import cfg
from src.utils.record_log import _logger
import json
import numpy as np
from src.utils.tree.str_transform import recursive_build_binary, \
    tokenize_str_format_tree
from src.utils.tree.tree2parent import transform_tree_to_parent_index
from src.utils.tree.shift_reduce import shift_reduce_constituency_forest
from src.utils.nlp import dynamic_length, dynamic_keep
from src.utils.file import load_glove, save_file
from tqdm import tqdm
import random
import math
import re

class Dataset(object):
    def __init__(self, data_file_path, data_type, dicts=None):
        self.data_type = data_type
        _logger.add('building data set object for %s' % data_type)
        assert data_type in ['train', 'dev', 'test']
        # check
        if data_type in ['dev', 'test']:
            assert dicts is not None

        # temporary params
        self.only_bi_tree = True

        raw_data = self.load_snli_data(data_file_path, data_type)
        data_with_tree = self.transform_str_to_tree(raw_data, data_type)
        # data_with_tree = self.generate_tree_shift_reduce_info(data_with_tree, data_type)
        processed_data_list = self.process_raw_data(data_with_tree, data_type)

        if data_type == 'train':
            self.dicts, self.max_lens = self.count_data_and_build_dict(processed_data_list)
        else:
            _, self.max_lens = self.count_data_and_build_dict(processed_data_list, False)
            self.dicts = dicts
        digital_data = self.digitize_data(processed_data_list, self.dicts, data_type)
        self.nn_data = self.clip_filter_data(digital_data, cfg.data_clip_method, data_type)
        self.sample_num = len(self.nn_data)
        if data_type == 'train':
            self.emb_mat_token, self.emb_mat_glove = self.generate_index2vec_matrix()


    # --------------------- internal use ---------------------
    def save_dict(self, path):
        save_file(self.dicts, path,'token and char dict data', 'pickle')

    def filter_data(self, data_type=None):
        data_type = data_type or self.data_type
        new_nn_data = []
        for sample in self.nn_data:
            if not sample['gold_label'] == '-':
                if data_type != 'train' or (len(sample['sentence1_token']) <= self.max_lens['sent'] and
                                                    len(sample['sentence2_token']) <= self.max_lens['sent']):
                    new_nn_data.append(sample)
        self.nn_data = new_nn_data
        self.sample_num = len(new_nn_data)

    def generate_batch_sample_iter(self, max_step=None):
        if max_step is not None:
            batch_size = cfg.train_batch_size

            def data_queue(data, batch_size):
                assert len(data) >= batch_size
                random.shuffle(data)
                data_ptr = 0
                dataRound = 0
                idx_b = 0
                step = 0
                while True:
                    if data_ptr + batch_size <= len(data):
                        yield data[data_ptr:data_ptr + batch_size], dataRound, idx_b
                        data_ptr += batch_size
                        idx_b += 1
                        step += 1
                    elif data_ptr + batch_size > len(data):
                        offset = data_ptr + batch_size - len(data)
                        out = data[data_ptr:]
                        random.shuffle(data)
                        out += data[:offset]
                        data_ptr = offset
                        dataRound += 1
                        yield out, dataRound, 0
                        idx_b = 1
                        step += 1
                    if step >= max_step:
                        break
            batch_num = math.ceil(len(self.nn_data) / batch_size)
            for sample_batch, data_round, idx_b in data_queue(self.nn_data, batch_size):
                yield sample_batch, batch_num, data_round, idx_b
        else:
            batch_size = cfg.test_batch_size
            batch_num = math.ceil(len(self.nn_data) / batch_size)
            idx_b = 0
            sample_batch = []
            for sample in self.nn_data:
                sample_batch.append(sample)
                if len(sample_batch) == batch_size:
                    yield sample_batch, batch_num, 0, idx_b
                    idx_b += 1
                    sample_batch = []
            if len(sample_batch) > 0:
                yield sample_batch, batch_num, 0, idx_b


    # --------------------- internal use ---------------------
    def load_snli_data(self, data_path, data_type):
        _logger.add()
        _logger.add('load file for %s' % data_type)
        dataset = []
        with open(data_path, 'r', encoding='utf-8') as file:
            for line in file:
                json_obj = json.loads(line)
                dataset.append(json_obj)
        _logger.done()
        return dataset

    def transform_str_to_tree(self, dataset, data_type):
        _logger.add()
        _logger.add('transforming str format tree into real tree for %s' % data_type)
        for sample in tqdm(dataset):
            sample['sentence1_binary_parse_tree'] = recursive_build_binary(
                tokenize_str_format_tree(sample['sentence1_binary_parse']))
            sample['sentence2_binary_parse_tree'] = recursive_build_binary(
                tokenize_str_format_tree(sample['sentence2_binary_parse']))
            # sample['sentence1_parse_tree'] = recursive_build_penn_format(
            #     tokenize_str_format_tree(sample['sentence1_parse']))
            # sample['sentence2_parse_tree'] = recursive_build_penn_format(
            #     tokenize_str_format_tree(sample['sentence2_parse']))

            # to node_list
            sample['sentence1_binary_parse_tree'], sample['sentence1_binary_parse_node_list'] = \
                transform_tree_to_parent_index(sample['sentence1_binary_parse_tree'])
            sample['sentence2_binary_parse_tree'], sample['sentence2_binary_parse_node_list'] = \
                transform_tree_to_parent_index(sample['sentence2_binary_parse_tree'])
            # sample['sentence1_parse_tree'], sample['sentence1_parse_node_list'] = \
            #     transform_tree_to_parent_index(sample['sentence1_parse_tree'])
            # sample['sentence2_parse_tree'], sample['sentence2_parse_node_list'] = \
            #     transform_tree_to_parent_index(sample['sentence2_parse_tree'])

            # shift reduce info
            # # s1
            s1_child_parent_node_indices = [(new_tree_node.node_index, new_tree_node.parent_index)
                                            for new_tree_node in sample['sentence1_binary_parse_node_list']]
            s1_sr = shift_reduce_constituency_forest(s1_child_parent_node_indices)
            s1_op_list, s1_node_list_in_stack, s1_reduce_mat = zip(*s1_sr)
            s1_sr_info = {'op_list': s1_op_list,
                          'reduce_mat': s1_reduce_mat,
                          'node_list_in_stack': s1_node_list_in_stack}
            sample['s1_sr_info'] = s1_sr_info

            # # s2
            s2_child_parent_node_indices = [(new_tree_node.node_index, new_tree_node.parent_index)
                                            for new_tree_node in sample['sentence2_binary_parse_node_list']]
            s2_sr = shift_reduce_constituency_forest(s2_child_parent_node_indices)
            s2_op_list, s2_node_list_in_stack, s2_reduce_mat = zip(*s2_sr)
            s2_sr_info = {'op_list': s2_op_list,
                          'reduce_mat': s2_reduce_mat,
                          'node_list_in_stack': s2_node_list_in_stack}
            sample['s2_sr_info'] = s2_sr_info

        _logger.done()
        return dataset

    def generate_tree_shift_reduce_info(self, dataset, data_type):
        _logger.add()
        _logger.add('generating tree shift reduce for %s' % data_type)

        for sample in dataset:
            # sent1
            s1_child_parent_node_indices = [(new_tree_node.node_index, new_tree_node.parent_index)
                                            for new_tree_node in sample['sentence1_binary_parse_node_list']]
            s1_sr = shift_reduce_constituency_forest(s1_child_parent_node_indices)
            s1_op_list, s1_node_list_in_stack, s1_reduce_mat = zip(*s1_sr)
            s1_sr_info = {'op_list': s1_op_list,
                          'reduce_mat': s1_reduce_mat,
                          'node_list_in_stack': s1_node_list_in_stack}
            sample['s1_sr_info'] = s1_sr_info
            # tree tag
            # s1_tree_tag = []
            # for node_idx in s1_node_list_in_stack:
            #     ### find tree node
            #     tree_node_found = None
            #     for tree_node in sample['sentence1_parse_node_list']:
            #         if tree_node.node_index == node_idx:
            #             tree_node_found = tree_node
            #             break
            #     assert tree_node_found is not None
            #     s1_tree_tag.append(tree_node_found.tag)
            # sample['s1_tree_tag'] = s1_tree_tag

            # s2
            s2_child_parent_node_indices = [(new_tree_node.node_index, new_tree_node.parent_index)
                                            for new_tree_node in sample['sentence2_binary_parse_node_list']]
            s2_sr = shift_reduce_constituency_forest(s2_child_parent_node_indices)
            s2_op_list, s2_node_list_in_stack, s2_reduce_mat = zip(*s2_sr)
            s2_sr_info = {'op_list': s2_op_list,
                          'reduce_mat': s2_reduce_mat,
                          'node_list_in_stack': s2_node_list_in_stack}
            sample['s2_sr_info'] = s2_sr_info
            # # tree tag
            # s2_tree_tag = []
            # for node_idx in s2_node_list_in_stack:
            #     ### find tree node
            #     tree_node_found = None
            #     for tree_node in sample['sentence2_parse_node_list']:
            #         if tree_node.node_index == node_idx:
            #             tree_node_found = tree_node
            #             break
            #     assert tree_node_found is not None
            #     s2_tree_tag.append(tree_node_found.tag)
            # sample['s2_tree_tag'] = s2_tree_tag

        return dataset

    def process_raw_data(self, dataset, data_type):
        def further_tokenize(temp_tokens):
            tokens = []  # [[(s,e),...],...]
            for token in temp_tokens:
                l = (
                "-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
                tokens.extend(re.split("([{}])".format("".join(l)), token))
            return tokens

        # tokens
        _logger.add()
        _logger.add('processing raw data for %s' % data_type)

        for sample in tqdm(dataset):
            sample['sentence1_token'] = [node.token
                                         for node in sample['sentence1_binary_parse_node_list'] if node.is_leaf]
            sample['sentence1_tag'] = [node.tag
                                       for node in sample['sentence1_binary_parse_node_list'] if node.is_leaf]

            sample['sentence2_token'] = [node.token
                                         for node in sample['sentence2_binary_parse_node_list'] if node.is_leaf]
            sample['sentence2_tag'] = [node.tag
                                       for node in sample['sentence2_binary_parse_node_list'] if node.is_leaf]

            if cfg.data_clip_method == 'no_tree':
                sample['sentence1_token'] = further_tokenize(sample['sentence1_token'])
                sample['sentence2_token'] = further_tokenize(sample['sentence2_token'])
        _logger.done()
        return dataset

    def count_data_and_build_dict(self, data_list, gene_dicts=True):
        def add_ept_and_unk(a_list):
            a_list.insert(0, '@@@empty')
            a_list.insert(1, '@@@unk')
            return a_list

        _logger.add()
        _logger.add('counting and build dictionaries')
        token_collection = []
        char_collection = []

        sent_len_collection = []
        token_len_collection = []

        for sample in data_list:
            token_collection += sample['sentence1_token'] + sample['sentence2_token']
            sent_len_collection += [len(sample['sentence1_token']), len(sample['sentence2_token'])]

            for token in sample['sentence1_token'] + sample['sentence2_token']:
                char_collection += list(token)
                token_len_collection.append(len(token))

        max_sent_len = dynamic_length(sent_len_collection, cfg.sent_len_rate, security=False)[0]
        max_token_len = dynamic_length(token_len_collection, 0.99, security=False)[0]
        if gene_dicts:
            tokenSet = dynamic_keep(token_collection, 1)
            charSet = dynamic_keep(char_collection, 1)

            if cfg.use_glove_unk_token:
                gloveData = load_glove(cfg.word_embedding_length)
                gloveTokenSet = list(gloveData.keys())
                if cfg.lower_word:
                    tokenSet = list(set([token.lower() for token in tokenSet]))  ##!!!
                    gloveTokenSet = list(set([token.lower() for token in gloveTokenSet]))  ##!!!

                # delete token from gloveTokenSet which appears in tokenSet
                for token in tokenSet:
                    try:
                        gloveTokenSet.remove(token)
                    except ValueError:
                        pass
            else:
                if cfg.lower_word:
                    tokenSet = list(set([token.lower() for token in tokenSet]))
                gloveTokenSet = []
            tokenSet = add_ept_and_unk(tokenSet)
            charSet = add_ept_and_unk(charSet)
            dicts = {'token': tokenSet, 'char': charSet, 'glove': gloveTokenSet}
        else:
            dicts = {}
        _logger.done()
        return dicts, {'sent': max_sent_len, 'token': max_token_len}

    def digitize_data(self, data_list, dicts, data_type):
        token2index = dict([(token, idx) for idx, token in enumerate(dicts['token'] + dicts['glove'])])
        char2index = dict([(token, idx) for idx, token in enumerate(dicts['char'])])

        def digitize_token(token):
            token = token if not cfg.lower_word else token.lower()
            try:
                return token2index[token]
            except KeyError:
                return 1

        def digitize_char(char):
            try:
                return char2index[char]
            except KeyError:
                return 1

        _logger.add()
        _logger.add('digitizing data: %s...' % data_type)
        for sample in tqdm(data_list):
            # sentence1/2_token/tag<list of str>
            sample['sentence1_token_digital'] = [digitize_token(token) for token in sample['sentence1_token']]
            sample['sentence2_token_digital'] = [digitize_token(token) for token in sample['sentence2_token']]

            sample['sentence1_char_digital'] = [[digitize_char(char) for char in list(token)]
                                                for token in sample['sentence1_token']]

            sample['sentence2_char_digital'] = [[digitize_char(char) for char in list(token)]
                                                for token in sample['sentence2_token']]
        _logger.done()
        return data_list

    def clip_filter_data(self, data_list, data_clip_method, data_type):
        _logger.add()
        _logger.add('%s cliping data for  %s...' % (data_clip_method, data_type))

        for sample in data_list:
            if data_clip_method == 'no_tree':
                sample.pop('sentence1_parse')
                sample.pop('sentence2_parse')
                # sample.pop('sentence1_parse_tree')
                # sample.pop('sentence2_parse_tree')
                # sample.pop('sentence1_parse_node_list')
                # sample.pop('sentence2_parse_node_list')
                sample.pop('sentence1_binary_parse')
                sample.pop('sentence2_binary_parse')
                sample.pop('sentence1_binary_parse_tree')
                sample.pop('sentence2_binary_parse_tree')
                sample.pop('sentence1_binary_parse_node_list')
                sample.pop('sentence2_binary_parse_node_list')
                sample.pop('s1_sr_info')
                sample.pop('s2_sr_info')
                # sample.pop('s1_tree_tag')
                # sample.pop('s2_tree_tag')
            elif data_clip_method == 'no_redundancy':
                sample.pop('sentence1_parse')
                sample.pop('sentence2_parse')
                # sample.pop('sentence1_parse_tree')
                # sample.pop('sentence2_parse_tree')
                # sample.pop('sentence1_parse_node_list')
                # sample.pop('sentence2_parse_node_list')

                sample.pop('sentence1_binary_parse')
                sample.pop('sentence2_binary_parse')
                sample.pop('sentence1_binary_parse_tree')
                sample.pop('sentence2_binary_parse_tree')

                for node in sample['sentence1_binary_parse_node_list']:
                    node.children_nodes = None
                    node.leaf_node_index_seq = None

                for node in sample['sentence1_binary_parse_node_list']:
                    node.children_nodes = None
                    node.leaf_node_index_seq = None


            else:
                raise AttributeError('no data clip method named as %s' % data_clip_method)
        _logger.done()
        return data_list

    def generate_index2vec_matrix(self):
        _logger.add()
        _logger.add('generate index to vector numpy matrix')

        token2vec = load_glove(cfg.word_embedding_length)
        if cfg.lower_word:
            newToken2vec = {}
            for token, vec in token2vec.items():
                newToken2vec[token.lower()] = vec
            token2vec = newToken2vec

        # prepare data from trainDataset and devDataset
        mat_token = np.random.uniform(-0.05, 0.05, size=(len(self.dicts['token']), cfg.word_embedding_length)).astype(
            cfg.floatX)

        mat_glove = np.zeros((len(self.dicts['glove']), cfg.word_embedding_length), dtype=cfg.floatX)

        for idx, token in enumerate(self.dicts['token']):
            try:
                mat_token[idx] = token2vec[token]
            except KeyError:
                pass
            mat_token[0] = np.zeros(shape=(cfg.word_embedding_length,), dtype=cfg.floatX)

        for idx, token in enumerate(self.dicts['glove']):
            mat_glove[idx] = token2vec[token]

        _logger.add('Done')
        return mat_token, mat_glove



if __name__ == '__main__':
    # print('train dataset class')
    # snli_train_path = '/Users/tshen/Workspaces/dataset/snli_1.0/snli_1.0_train.jsonl'
    # train_dataset = Dataset(snli_train_path, 'train')
    # print('dev dataset class')
    # snli_dev_path = '/Users/tshen/Workspaces/dataset/snli_1.0/snli_1.0_dev.jsonl'
    # dev_dataset = Dataset(snli_dev_path, 'train')
    print('test dataset class')
    snli_test_path = '/home/tshen/Workspaces/dataset/snli_1.0/snli_1.0_test.jsonl'
    test_dataset = Dataset(snli_test_path, 'train')








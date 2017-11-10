from configs import cfg
from src.utils.record_log import _logger
from src.utils.file import load_glove, save_file
from src.utils.nlp import dynamic_length, dynamic_keep
from src.utils.tree.shift_reduce import shift_reduce_constituency_forest
from os.path import join as pjoin
import random
import math
import numpy as np


class Dataset(object):
    def __init__(self, data_list, data_type, dicts= None):
        self.data_type = data_type
        _logger.add('building data set object for %s' % data_type)
        assert data_type in ['train', 'dev', 'test']
        # check
        if data_type in ['dev', 'test']:
            assert dicts is not None

        processed_data_list = self.process_raw_data(data_list, data_type)

        if data_type == 'train':
            self.dicts, self.max_lens = self.count_data_and_build_dict(processed_data_list)
        else:
            _, self.max_lens = self.count_data_and_build_dict(processed_data_list, False)
            self.dicts = dicts

        self.digitized_data_list = self.digitize_dataset(processed_data_list, self.dicts, data_type)
        self.nn_data, self.sample_num = self.gene_sub_trees_and_shift_reduce_info()

        self.emb_mat_token, self.emb_mat_glove = None, None
        if data_type == 'train':
            self.emb_mat_token, self.emb_mat_glove = self.generate_index2vec_matrix()

    # external_use
    # how to generate sub tree? : done find all node belonging to the subtree
    def save_dict(self,path):
        save_file(self.dicts, path,'token and char dict data', 'pickle')

    def generate_batch_sample_iter(self, max_step=None):
        nn_data_list = []
        for trees in self.nn_data:
            for tree in trees:
                nn_data_list.append(tree)

        if max_step is not None:
            # todo: for data imbalance
            if cfg.data_imbalance and not cfg.only_sentence:
                add_data_list = []
                for sample in nn_data_list:
                    sentiment_float = sample['root_node']['sentiment_label']
                    if sentiment_float<=0.4 or sentiment_float>0.6 or sample['is_sent']:
                        add_data_list += [sample] * 3
                nn_data_list += add_data_list
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

            batch_num = math.ceil(len(nn_data_list) / batch_size)
            for sample_batch, data_round, idx_b in data_queue(nn_data_list, batch_size):
                yield sample_batch, batch_num, data_round, idx_b
        else:
            batch_size = cfg.test_batch_size
            batch_num = math.ceil(len(nn_data_list) / batch_size)
            idx_b = 0
            sample_batch = []
            for sample in nn_data_list:
                sample_batch.append(sample)
                if len(sample_batch) == batch_size:
                    yield sample_batch, batch_num, 0, idx_b
                    idx_b += 1
                    sample_batch = []
            if len(sample_batch) > 0:
                yield sample_batch, batch_num, 0, idx_b

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
        mat_token = np.random.normal(0, 1, size=(len(self.dicts['token']), cfg.word_embedding_length)).astype(
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

                # internal use

    def filter_data(self, only_sent=False, fine_grained=False):
        _logger.add()
        _logger.add('filtering data for %s, only sentence: %s' % (self.data_type, only_sent))
        if only_sent:
            counter = 0
            new_nn_data = []
            for trees in self.nn_data:
                new_trees = []
                for tree in trees:
                    if tree['is_sent']:
                        new_trees.append(tree)
                        counter += 1
                new_nn_data.append(new_trees)
            self.nn_data = new_nn_data
            self.sample_num = counter

        if not fine_grained:
            # delete the neutral sample
            counter = 0
            new_nn_data = []
            for trees in self.nn_data:
                new_trees = []
                for tree in trees:
                    sent_label = tree['root_node']['sentiment_label']
                    if sent_label <= 0.4 or sent_label > 0.6:
                        counter += 1
                        new_trees.append(tree)
                new_nn_data.append(new_trees)
            self.nn_data = new_nn_data
            self.sample_num = counter



    # internal use
    def process_raw_data(self, data_list, data_type):
        _logger.add()
        _logger.add('processing raw data: %s...' % data_type)
        for sample in data_list:
            for tree_node in sample:
                # node_index, parent_index, token_seq, leaf_node_index_seq, is_leaf, token, sentiment_label
                # char_seq
                tree_node['char_seq'] = [list(token) for token in tree_node['token_seq']]
        _logger.done()
        return data_list

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
            for tree_node in sample:
                token_collection += tree_node['token_seq']
                sent_len_collection.append(len(tree_node['token_seq']))
                for char_seq in tree_node['char_seq']:
                    char_collection += char_seq
                    token_len_collection.append(len(char_seq))

        max_sent_len = dynamic_length(sent_len_collection, 1, security=False)[0]
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

    def digitize_dataset(self, data_list, dicts, data_type):
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
        for sample in data_list:
            for tree_node in sample:
                tree_node['token_seq_digital'] = [digitize_token(token) for token in tree_node['token_seq']]
                tree_node['char_seq_digital'] = [[digitize_char(char) for char in char_seq]
                                                 for char_seq in tree_node['char_seq']]
        _logger.done()
        return data_list

    def gene_sub_trees_and_shift_reduce_info(self):
        _logger.add()
        _logger.add('generating sub-trees and shift reduce info: %s...' % self.data_type)
        counter = 0
        new_data_list = []
        for tree in self.digitized_data_list:
            sub_trees = []
            idx_to_node_dict = dict((tree_node['node_index'], tree_node)
                                    for tree_node in tree)
            for tree_node in tree:
                # get all node for a sub tree
                if tree_node['is_leaf']:
                    new_sub_tree = [tree_node]
                else:
                    new_sub_tree = []
                    new_sub_tree_leaves = [idx_to_node_dict[node_index]
                                           for node_index in tree_node['leaf_node_index_seq']]
                    new_sub_tree += new_sub_tree_leaves
                    for leaf_node in new_sub_tree_leaves:
                        pre_node = leaf_node
                        while pre_node['parent_index'] > 0 and pre_node != tree_node: # fixme
                            cur_node = idx_to_node_dict[pre_node['parent_index']]
                            if cur_node not in new_sub_tree:
                                new_sub_tree.append(cur_node)
                            pre_node = cur_node
                # get shift reduce info
                child_node_indices = [new_tree_node['node_index'] for new_tree_node in new_sub_tree]
                parent_node_indices = [new_tree_node['parent_index']
                                       if new_tree_node['parent_index'] in child_node_indices else 0
                                       for new_tree_node in new_sub_tree]
                sr_result = shift_reduce_constituency_forest(list(zip(child_node_indices, parent_node_indices)))
                operation_list, node_list_in_stack, reduce_mat = zip(*sr_result)
                shift_reduce_info = {'op_list': operation_list,
                                     'reduce_mat': reduce_mat,
                                     'node_list_in_stack': node_list_in_stack}
                sub_tree = {'tree_nodes': new_sub_tree, 'shift_reduce_info': shift_reduce_info,
                            'root_node': tree_node, 'is_sent': True if tree_node['parent_index'] == 0 else False
                            }
                sub_trees.append(sub_tree)
                counter += 1
            new_data_list.append(sub_trees)
        return new_data_list, counter


class RawDataProcessor(object):
    def __init__(self, data_dir):
        self.trees, self.dictionary, self.sentiment_labels, self.dataset_split = self.load_sst_data(data_dir)

    # external use
    def get_data_list(self, data_type):
        if data_type == 'train':
            target = 1
        elif data_type == 'test':
            target = 2
        else:
            target = 3

        data_list = []
        for _type, tree in zip(self.dataset_split, self.trees):
            if _type == target:
                data_list.append(tree)

        return data_list

    # internal use
    def load_sst_data(self, data_dir):
        # dictionary
        dictionary = {}
        with open(pjoin(data_dir, 'dictionary.txt'), encoding='utf-8') as file:
            for line in file:
                line = line.strip().split('|')
                assert len(line) == 2
                dictionary[line[0]] = int(line[1])

        # sentiment_labels
        sentiment_labels = {}
        with open(pjoin(data_dir, 'sentiment_labels.txt'), encoding='utf-8') as file:
            file.readline()  # for table head
            for line in file:
                line = line.strip().split('|')
                sent_float_value = float(line[1])

                sentiment_labels[int(line[0])] = sent_float_value

        # STree.txt and SOStr.txt
        trees = []
        with open(pjoin(data_dir, 'STree.txt'), encoding='utf-8') as file_STree, \
                open(pjoin(data_dir, 'SOStr.txt'), encoding='utf-8') as file_SOStr:
            for STree, SOStr in zip(file_STree, file_SOStr):
                sent_tree = []
                STree = list(map(int, STree.strip().split('|')))
                SOStr = SOStr.strip().split('|')

                for idx_t, parent_idx in enumerate(STree):
                    try:
                        token = SOStr[idx_t]
                        is_leaf = True
                        leaf_node_index_seq = [idx_t+1]
                    except IndexError:
                        token = ''
                        is_leaf = False
                        leaf_node_index_seq = []

                    new_node = {'node_index': idx_t+1, 'parent_index': parent_idx,
                                'token': token, 'is_leaf': is_leaf,
                                'leaf_node_index_seq': leaf_node_index_seq, }
                    sent_tree.append(new_node)

                # update leaf_node_index_seq
                idx_to_node_dict = dict((tree_node['node_index'], tree_node)
                                        for tree_node in sent_tree)
                for tree_node in sent_tree:
                    if not tree_node['is_leaf']: break
                    pre_node = tree_node
                    while pre_node['parent_index'] > 0:
                        cur_node = idx_to_node_dict[pre_node['parent_index']]
                        cur_node['leaf_node_index_seq'] += pre_node['leaf_node_index_seq']
                        cur_node['leaf_node_index_seq'] = list(
                            sorted(list(set(cur_node['leaf_node_index_seq']))))
                        pre_node = cur_node

                # update sentiment and add token_seq
                for tree_node in sent_tree:
                    tokens = [sent_tree[node_idx-1]['token'] for node_idx in tree_node['leaf_node_index_seq']]
                    phrase = ' '.join(tokens)
                    tree_node['sentiment_label'] = sentiment_labels[dictionary[phrase]]
                    tree_node['token_seq'] = tokens

                trees.append(sent_tree)

        # dataset_split (head)
        dataset_split = []
        with open(pjoin(data_dir, 'datasetSplit.txt'), encoding='utf-8') as file:
            file.readline()  # for table head
            for line in file:
                dataset_split.append(int(line.strip().split(',')[1]))

        return trees, dictionary, sentiment_labels, dataset_split



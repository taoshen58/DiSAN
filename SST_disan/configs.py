import argparse
import os
from os.path import join
from src.utils.time_counter import TimeCounter

class Configs(object):
    def __init__(self):
        self.project_dir = os.getcwd()
        self.dataset_dir = join(self.project_dir, 'dataset')

        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()
        parser.register('type', 'bool', (lambda x: x.lower() in ('True', "yes", "true", "t", "1")))

        # @ ----- control ----
        parser.add_argument('--debug', type='bool', default=False, help='whether run as debug mode')
        parser.add_argument('--mode', type=str, default='train', help='train, dev or test')
        parser.add_argument('--network_type', type=str, default='test', help='network type')
        parser.add_argument('--fine_grained', type='bool', default=True, help='5 classes or 2 classes [True|False]')
        parser.add_argument('--only_sentence', type='bool', default=False, help='sentence or phrase level')
        parser.add_argument('--data_imbalance', type='bool', default=True, help='balance data distribution')
        parser.add_argument('--log_period', type=int, default=500, help='save tf summary period')  ###  change for running
        parser.add_argument('--save_period', type=int, default=3000, help='abandoned')
        parser.add_argument('--eval_period', type=int, default=1000, help='evaluation period')  ###  change for running
        parser.add_argument('--gpu', type=int, default=3, help='employed gpu index')
        parser.add_argument('--gpu_mem', type=float, default=0.96, help='gpu memory ratio to employ')
        parser.add_argument('--model_dir_suffix', type=str, default='', help='model folder name suffix')
        parser.add_argument('--swap_memory', type='bool', default=False, help='abandoned')
        parser.add_argument('--load_model', type='bool', default=False, help='do not use')
        parser.add_argument('--load_step', type=int, default=None, help='do not use')
        parser.add_argument('--load_path', type=str, default=None, help='specify which pre-trianed model to be load')

        # @ ----------training ------
        parser.add_argument('--max_epoch', type=int, default=200, help='max epoch number')
        parser.add_argument('--num_steps', type=int, default=120000, help='max steps num')
        parser.add_argument('--train_batch_size', type=int, default=64, help='Train Batch Size')
        parser.add_argument('--test_batch_size', type=int, default=128, help='Test Batch Size')
        parser.add_argument('--optimizer', type=str, default='adadelta', help='choose an optimizer[adadelta|adam]')
        parser.add_argument('--learning_rate', type=float, default=0.5, help='Init Learning rate')
        parser.add_argument('--wd', type=float, default=1e-4, help='weight decay factor/l2 decay factor')
        parser.add_argument('--var_decay', type=float, default=0.999, help='Learning rate')  # ema
        parser.add_argument('--decay', type=float, default=0.9, help='summary decay')  # ema

        # @ ----- Text Processing ----
        parser.add_argument('--word_embedding_length', type=int, default=300, help='word embedding length')
        parser.add_argument('--glove_corpus', type=str, default='6B', help='choose glove corpus to employ')
        parser.add_argument('--use_glove_unk_token', type='bool', default=True, help='')
        parser.add_argument('--lower_word', type='bool', default=True, help='')

        # @ ------neural network-----
        parser.add_argument('--use_char_emb', type='bool', default=False, help='abandoned')
        parser.add_argument('--use_token_emb', type='bool', default=True, help='abandoned')
        parser.add_argument('--char_embedding_length', type=int, default=8, help='abandoned')
        parser.add_argument('--char_out_size', type=int, default=150, help='abandoned')
        parser.add_argument('--out_channel_dims', type=str, default='50,50,50', help='abandoned')
        parser.add_argument('--filter_heights', type=str, default='1,3,5', help='abandoned')
        parser.add_argument('--highway_layer_num', type=int, default=2, help='highway layer number(abandoned)')

        parser.add_argument('--dropout', type=float, default=0.7, help='dropout keep prob')
        parser.add_argument('--hidden_units_num', type=int, default=300, help='Hidden units number of Neural Network')
        parser.add_argument('--fine_tune', type='bool', default=False, help='(abandoned, keep False)')  # ema

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))

        # ------- name --------
        self.processed_name = 'processed' + self.get_params_str(['lower_word', 'use_glove_unk_token',
                                                                 'glove_corpus', 'word_embedding_length']) + '.pickle'
        self.dict_name = 'dicts' + self.get_params_str(['lower_word', 'use_glove_unk_token',
                                                        ])

        if not self.network_type == 'test':
            params_name_list = ['network_type', 'fine_grained', 'data_imbalance',
                                'only_sentence','dropout', 'word_embedding_length',
                                'char_out_size', 'hidden_units_num',  'learning_rate',
                                'wd', 'optimizer']
            if self.network_type.startswith('baseline'):
                params_name_list.append('tree_hn')
                params_name_list.append('shift_reduce_method')
                params_name_list.append('')
            if self.network_type.startswith('emb_direct_attn') or \
                    self.network_type.startswith('emb_interact_attn'):
                params_name_list.append('method_index')
                params_name_list.append('use_bi')
            self.model_name = self.get_params_str(params_name_list)

        else:
            self.model_name = self.network_type
        self.model_ckpt_name = 'modelfile.ckpt'



        # ---------- dir -------------
        self.data_dir = join(self.dataset_dir, 'stanfordSentimentTreebank')
        self.glove_dir = join(self.dataset_dir, 'glove')
        self.result_dir = self.mkdir(self.project_dir, 'result')
        self.standby_log_dir = self.mkdir(self.result_dir, 'log')
        self.dict_dir = self.mkdir(self.result_dir, 'dict')
        self.processed_dir = self.mkdir(self.result_dir, 'processed_data')

        self.log_dir = None
        self.all_model_dir = self.mkdir(self.result_dir, 'model')
        self.model_dir = self.mkdir(self.all_model_dir, self.model_dir_suffix + self.model_name)
        self.log_dir = self.mkdir(self.model_dir, 'log_files')
        self.summary_dir = self.mkdir(self.model_dir, 'summary')
        self.ckpt_dir = self.mkdir(self.model_dir, 'ckpt')
        self.answer_dir = self.mkdir(self.model_dir, 'answer')

        # -------- path --------
        self.processed_path = join(self.processed_dir, self.processed_name)
        self.dict_path = join(self.dict_dir, self.dict_name)
        self.ckpt_path = join(self.ckpt_dir, self.model_ckpt_name)

        self.extre_dict_path = join(self.dict_dir, 'extra_dict.json')

        # dtype
        self.floatX = 'float32'
        self.intX = 'int32'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        self.time_counter = TimeCounter()

    def get_params_str(self, params):
        def abbreviation(name):
            words = name.strip().split('_')
            abb = ''
            for word in words:
                abb += word[0]
            return abb

        abbreviations = map(abbreviation, params)
        model_params_str = ''
        for paramsStr, abb in zip(params, abbreviations):
            model_params_str += '_' + abb + '_' + str(eval('self.args.' + paramsStr))
        return model_params_str

    def mkdir(self, *args):
        dirPath = join(*args)
        if not os.path.exists(dirPath):
            os.makedirs(dirPath)
        return dirPath

    def get_file_name_from_path(self, path):
        assert isinstance(path, str)
        fileName = '.'.join((path.split('/')[-1]).split('.')[:-1])
        return fileName

    def sentiment_float_to_int(self, sentiment_float, fine_grained=None):
        fine_grained = None or self.fine_grained
        if fine_grained:
            if sentiment_float <= 0.2:
                sentiment_int = 0
            elif sentiment_float <= 0.4:
                sentiment_int = 1
            elif sentiment_float <= 0.6:
                sentiment_int = 2
            elif sentiment_float <= 0.8:
                sentiment_int = 3
            else:
                sentiment_int = 4
        else:
            if sentiment_float < 0.5:
                sentiment_int = 0
            else:
                sentiment_int = 1
        return sentiment_int

cfg = Configs()

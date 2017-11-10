import platform
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
        parser.register('type', 'bool', (lambda x: x.lower() in ("yes", "true", "t", "1")))

        # @ ----- control ----
        parser.add_argument('--debug', type='bool', default=False, help='whether run as debug mode')
        parser.add_argument('--mode', type=str, default='train', help='train, dev or test')
        parser.add_argument('--network_type', type=str, default='test', help='network type')
        parser.add_argument('--log_period', type=int, default=2000, help='save tf summary period')
        parser.add_argument('--save_period', type=int, default=3000, help='abandoned')
        parser.add_argument('--eval_period', type=int, default=500, help='evaluation period')
        parser.add_argument('--gpu', type=int, default=0, help='employed gpu index')
        parser.add_argument('--gpu_mem', type=float, default=0.96, help='gpu memory ratio to employ')
        parser.add_argument('--model_dir_suffix', type=str, default='', help='model folder name suffix')
        parser.add_argument('--swap_memory', type='bool', default=False, help='abandoned')
        parser.add_argument('--load_model', type='bool', default=False, help='do not use')
        parser.add_argument('--load_step', type=int, default=None, help='do not use')
        parser.add_argument('--load_path', type=str, default=None, help='specify which pre-trianed model to be load')

        # @ ----------training ------
        parser.add_argument('--max_epoch', type=int, default=100, help='max epoch number')
        parser.add_argument('--num_steps', type=int, default=400000, help='max steps num')
        parser.add_argument('--train_batch_size', type=int, default=64, help='Train Batch Size')
        parser.add_argument('--test_batch_size', type=int, default=100, help='Test Batch Size')
        parser.add_argument('--optimizer', type=str, default='adadelta', help='choose an optimizer[adadelta|adam]')
        parser.add_argument('--learning_rate', type=float, default=0.5, help='Init Learning rate')
        parser.add_argument('--dy_lr', type='bool', default=False, help='if decay lr during training')
        parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay')
        parser.add_argument('--dropout', type=float, default=0.75, help='dropout keep prob')
        parser.add_argument('--wd', type=float, default=5e-5, help='weight decay factor/l2 decay factor')
        parser.add_argument('--var_decay', type=float, default=0.999, help='Learning rate')  # ema
        parser.add_argument('--decay', type=float, default=0.9, help='summary decay')  # ema


        # @ ----- Text Processing ----
        parser.add_argument('--word_embedding_length', type=int, default=300, help='word embedding length')
        parser.add_argument('--glove_corpus', type=str, default='6B', help='choose glove corpus to employ')
        parser.add_argument('--use_glove_unk_token', type='bool', default=True, help='')
        parser.add_argument('--lower_word', type='bool', default=True, help='')
        parser.add_argument('--data_clip_method', type=str, default='no_tree',
                            help='for space-efficiency[no_tree|]no_redundancy')
        parser.add_argument('--sent_len_rate', type=float, default=0.97, help='delete too long sentences')

        # @ ------neural network-----
        parser.add_argument('--use_char_emb', type='bool', default=False, help='abandoned')
        parser.add_argument('--use_token_emb', type='bool', default=True, help='abandoned')
        parser.add_argument('--char_embedding_length', type=int, default=8, help='(abandoned)')
        parser.add_argument('--char_out_size', type=int, default=150, help='(abandoned)')
        parser.add_argument('--out_channel_dims', type=str, default='50,50,50', help='(abandoned)')
        parser.add_argument('--filter_heights', type=str, default='1,3,5', help='(abandoned)')
        parser.add_argument('--highway_layer_num', type=int, default=2, help='highway layer number(abandoned)')

        parser.add_argument('--hidden_units_num', type=int, default=300, help='Hidden units number of Neural Network')
        parser.add_argument('--tree_hn', type=int, default=100, help='(abandoned)')

        parser.add_argument('--fine_tune', type='bool', default=False, help='(abandoned, keep False)')  # ema

        # # emb_opt_direct_attn
        parser.add_argument('--batch_norm', type='bool', default=False, help='(abandoned, keep False)')
        parser.add_argument('--activation', type=str, default='relu', help='(abandoned')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))

        # ------- name --------
        self.train_data_name = 'snli_1.0_train.jsonl'
        self.dev_data_name = 'snli_1.0_dev.jsonl'
        self.test_data_name = 'snli_1.0_test.jsonl'

        self.processed_name = 'processed' + self.get_params_str(['lower_word', 'use_glove_unk_token',
                                                                 'glove_corpus', 'word_embedding_length',
                                                                 'sent_len_rate',
                                                                 'data_clip_method']) + '.pickle'
        self.dict_name = 'dicts' + self.get_params_str(['lower_word', 'use_glove_unk_token',
                                                        ])

        if not self.network_type == 'test':
            params_name_list = ['network_type', 'dropout', 'glove_corpus',
                                'word_embedding_length', 'fine_tune', 'char_out_size', 'sent_len_rate',
                                'hidden_units_num', 'wd', 'optimizer', 'learning_rate', 'dy_lr', 'lr_decay']
            self.model_name = self.get_params_str(params_name_list)
        else:
            self.model_name = self.network_type
        self.model_ckpt_name = 'modelfile.ckpt'

        # ---------- dir -------------
        self.data_dir = join(self.dataset_dir, 'snli_1.0')
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
        self.train_data_path = join(self.data_dir, self.train_data_name)
        self.dev_data_path = join(self.data_dir, self.dev_data_name)
        self.test_data_path = join(self.data_dir, self.test_data_name)

        self.processed_path = join(self.processed_dir, self.processed_name)
        self.dict_path = join(self.dict_dir, self.dict_name)
        self.ckpt_path = join(self.ckpt_dir, self.model_ckpt_name)

        self.extre_dict_path = join(self.dict_dir,
                                    'extra_dict'+self.get_params_str(['data_clip_method'])+'.json')

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


cfg = Configs()

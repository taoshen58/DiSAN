import math
import tensorflow as tf

from configs import cfg
from src.dataset import Dataset
from src.evaluator import Evaluator
from src.graph_handler import GraphHandler
from src.perform_recorder import PerformRecoder
from src.utils.file import load_file, save_file
from src.utils.record_log import _logger

# choose model
network_type = cfg.network_type
if network_type == 'exp_bi_lstm_mul_attn':  # check, running
    from src.model.exp_bi_lstm_mul_attn import ModelExpBiLSTMMulAttn as Model
elif network_type == 'exp_emb_attn':  # check, done
    from src.model.exp_emb_attn import ModelExpEmbAttn as Model
elif network_type == 'exp_emb_mul_attn':  # check, done
    from src.model.exp_emb_mul_attn import ModelExpEmbMulAttn as Model
elif network_type == 'exp_emb_self_mul_attn':
    from src.model.exp_emb_self_mul_attn import ModelExpEmbSelfMulAttn as Model
elif network_type == 'exp_emb_dir_mul_attn':
    from src.model.exp_emb_dir_mul_attn import ModelExpEmbDirMulAttn as Model
elif network_type == 'disan':
    from src.model.model_disan import ModelDiSAN as Model

model_type_set = ['exp_bi_lstm_mul_attn', 'exp_emb_attn', 'exp_emb_mul_attn',
                  'exp_emb_self_mul_attn', 'exp_emb_dir_mul_attn', 'disan']


def train():
    output_model_params()
    loadFile = True
    ifLoad, data = False, None
    if loadFile:
        ifLoad, data = load_file(cfg.processed_path, 'processed data', 'pickle')
    if not ifLoad or not loadFile:
        train_data_obj = Dataset(cfg.train_data_path, 'train')
        dev_data_obj = Dataset(cfg.dev_data_path, 'dev', dicts=train_data_obj.dicts)
        test_data_obj = Dataset(cfg.test_data_path, 'test', dicts=train_data_obj.dicts)

        save_file({'train_data_obj': train_data_obj, 'dev_data_obj': dev_data_obj, 'test_data_obj': test_data_obj},
                  cfg.processed_path)

        train_data_obj.save_dict(cfg.dict_path)
    else:
        train_data_obj = data['train_data_obj']
        dev_data_obj = data['dev_data_obj']
        test_data_obj = data['test_data_obj']

    train_data_obj.filter_data()
    dev_data_obj.filter_data()
    test_data_obj.filter_data()

    emb_mat_token, emb_mat_glove = train_data_obj.emb_mat_token, train_data_obj.emb_mat_glove

    with tf.variable_scope(network_type) as scope:
        if network_type in model_type_set:
            model = Model(emb_mat_token, emb_mat_glove, len(train_data_obj.dicts['token']),
                          len(train_data_obj.dicts['char']), train_data_obj.max_lens['token'], scope.name)
    graphHandler = GraphHandler(model)
    evaluator = Evaluator(model)
    performRecoder = PerformRecoder(3)

    if cfg.gpu_mem < 1.:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_mem,
                                    allow_growth=True)
    else:
        gpu_options = tf.GPUOptions()
    graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    # graph_config.gpu_options.allow_growth = True
    sess = tf.Session(config=graph_config)
    graphHandler.initialize(sess)

    # begin training
    steps_per_epoch = int(math.ceil(1.0 * train_data_obj.sample_num / cfg.train_batch_size))
    num_steps = cfg.num_steps or steps_per_epoch * cfg.max_epoch

    global_step = 0

    for sample_batch, batch_num, data_round, idx_b in train_data_obj.generate_batch_sample_iter(num_steps):
        global_step = sess.run(model.global_step) + 1
        if_get_summary = global_step % (cfg.log_period or steps_per_epoch) == 0
        loss, summary, train_op = model.step(sess, sample_batch, get_summary=if_get_summary)
        if global_step % 100 == 0 or global_step == 1:
            _logger.add('data round: %d: %d/%d, global step:%d -- loss: %.4f' %
                        (data_round, idx_b, batch_num, global_step, loss))

        if if_get_summary:
            graphHandler.add_summary(summary, global_step)

        # Occasional evaluation
        if global_step > int(cfg.num_steps - 100000) and (cfg.global_step % (cfg.eval_period or steps_per_epoch) == 0):
            # ---- dev ----
            dev_loss, dev_accu = evaluator.get_evaluation(
                sess, dev_data_obj, global_step
            )
            _logger.add('==> for dev, loss: %.4f, accuracy: %.4f' %
                        (dev_loss, dev_accu))
            # ---- test ----
            test_loss, test_accu = evaluator.get_evaluation(
                sess, test_data_obj, global_step
            )
            _logger.add('~~> for test, loss: %.4f, accuracy: %.4f' %
                        (test_loss, test_accu))

            model.update_learning_rate(dev_loss, cfg.lr_decay)
            is_in_top, deleted_step = performRecoder.update_top_list(global_step, dev_accu, sess)

        this_epoch_time, mean_epoch_time = cfg.time_counter.update_data_round(data_round)
        if this_epoch_time is not None and mean_epoch_time is not None:
            _logger.add('##> this epoch time: %f, mean epoch time: %f' % (this_epoch_time, mean_epoch_time))

    do_analyse_snli(_logger.path)


def test():

    assert cfg.load_path is not None
    output_model_params()
    loadFile = True
    ifLoad, data = False, None
    if loadFile:
        ifLoad, data = load_file(cfg.processed_path, 'processed data', 'pickle')
    if not ifLoad or not loadFile:
        raise RuntimeError('cannot find pre-processed dataset')
    else:
        train_data_obj = data['train_data_obj']
        dev_data_obj = data['dev_data_obj']
        test_data_obj = data['test_data_obj']

    train_data_obj.filter_data('test')
    dev_data_obj.filter_data('test')
    test_data_obj.filter_data('test')

    emb_mat_token, emb_mat_glove = train_data_obj.emb_mat_token, train_data_obj.emb_mat_glove

    with tf.variable_scope(network_type) as scope:
        if network_type in model_type_set:
            model = Model(emb_mat_token, emb_mat_glove, len(train_data_obj.dicts['token']),
                          len(train_data_obj.dicts['char']), train_data_obj.max_lens['token'], scope.name)
    graphHandler = GraphHandler(model)
    evaluator = Evaluator(model)

    if cfg.gpu_mem < 1.:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=cfg.gpu_mem,
                                    allow_growth=True)
    else:
        gpu_options = tf.GPUOptions()
    graph_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=graph_config)
    graphHandler.initialize(sess)

    # ---- dev ----
    dev_loss, dev_accu = evaluator.get_evaluation(
        sess, dev_data_obj, None
    )
    _logger.add('==> for dev, loss: %.4f, accuracy: %.4f' %
                (dev_loss, dev_accu))
    # ---- test ----
    test_loss, test_accu = evaluator.get_evaluation(
        sess, test_data_obj, None
    )
    _logger.add('~~> for test, loss: %.4f, accuracy: %.4f' %
                (test_loss, test_accu))

    train_loss, train_accu = evaluator.get_evaluation(
        sess, train_data_obj, None
    )
    _logger.add('--> for test, loss: %.4f, accuracy: %.4f' %
                (train_loss, train_accu))


def main(_):
    if cfg.mode == 'train':
        train()
    elif cfg.mode == 'test':
        test()
    else:
        raise RuntimeError('no running mode named as %s' % cfg.mode)


def output_model_params():
    _logger.add()
    _logger.add('==>model_title: ' + cfg.model_name[1:])
    _logger.add()
    for key,value in cfg.args.__dict__.items():
        if key not in ['test','shuffle']:
            _logger.add('%s: %s' % (key, value))


def do_analyse_snli(file_path, dev=True, use_loss=False, stop=None):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        find_entry = False
        output = [0, 0., 0., 0., 0.] # xx, dev, test,
        for line in file:
            if not find_entry:
                if line.startswith('data round'):  # get step
                    output[0] = int(line.split(' ')[-4].split(':')[-1])
                    if stop is not None and output[0] > stop: break
                if line.startswith('==> for dev'):  # dev
                    output[1] = float(line.split(' ')[-1])
                    output[2] = float(line.split(' ')[-3][:-1])
                    find_entry = True
            else:
                if line.startswith('~~> for test'):  # test
                    output[3] = float(line.split(' ')[-1])
                    output[4] = float(line.split(' ')[-3][:-1])
                    results.append(output)
                    find_entry = False
                    output = [0, 0., 0., 0., 0.]

    # max step
    if len(results) > 0:
        print('max step:', results[-1][0])

    # sort
    sort = 1 if dev else 3
    if use_loss: sort += 1
    output = list(sorted(results, key=lambda elem: elem[sort], reverse=not use_loss))

    for elem in output[:3]:
        print('step: %d, dev: %.4f, dev_loss: %.4f, test: %.4f, test_loss: %.4f' %
              (elem[0], elem[1], elem[2], elem[3],elem[4]))



if __name__ == '__main__':
    tf.app.run()




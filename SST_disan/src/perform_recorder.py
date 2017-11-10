from configs import cfg
import tensorflow as tf
import os


class PerformRecoder(object):
    def __init__(self, top_limit_num=3):
        self.top_limit_num = top_limit_num
        self.dev_top_list = []  # list of tuple(step, dev_accu)
        self.saver = tf.train.Saver(max_to_keep=None)

    def update_top_list(self, global_step, dev_accu, sess):
        cur_ckpt_path = self.ckpt_file_path_generator(global_step)
        self.dev_top_list.append([global_step, dev_accu])
        self.dev_top_list = list(sorted(self.dev_top_list, key=lambda elem: elem[1], reverse=True))

        if len(self.dev_top_list) <= self.top_limit_num:
            self.create_ckpt_file(sess, cur_ckpt_path)
            return True, None
        elif len(self.dev_top_list) == self.top_limit_num + 1:
            out_state = self.dev_top_list[-1]
            self.dev_top_list = self.dev_top_list[:-1]
            if out_state[0] == global_step:
                return False, None
            else:  # add and delete
                self.delete_ckpt_file(self.ckpt_file_path_generator(out_state[0]))
                self.create_ckpt_file(sess, cur_ckpt_path)
                return True, out_state[0]

        else:
            raise RuntimeError()

    def ckpt_file_path_generator(self, step):
        return os.path.join(cfg.ckpt_dir, 'top_result_saver_step_%d.ckpt' % step)

    def delete_ckpt_file(self, ckpt_file_path):
        if os.path.isfile(ckpt_file_path+'.meta'):
            os.remove(ckpt_file_path+'.meta')
        if os.path.isfile(ckpt_file_path+'.index'):
            os.remove(ckpt_file_path+'.index')
        if os.path.isfile(ckpt_file_path+'.data-00000-of-00001'):
            os.remove(ckpt_file_path+'.data-00000-of-00001')

    def create_ckpt_file(self, sess, ckpt_file_path):
        self.saver.save(sess, ckpt_file_path)













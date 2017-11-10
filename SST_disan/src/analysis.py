from configs import cfg
from src.utils.file import load_file
import nltk
import csv, os
import math


class OutputAnalysis(object):
    def __init__(self):
        # todo:
        # 1. all sample classification distribution and their accuracy
        # 1. all sentence sample classification distribution and their accuracy
        # 3. output all sentence sample with label and prediction
        pass

    @staticmethod
    def do_analysis(dataset_obj, pred_arr,eval_arr, save_dir, fine_grained=True):
        out_class_num = 5 if fine_grained else 2

        save_dir = cfg.mkdir(save_dir, dataset_obj.data_type)
        sample_list = []
        int_labels = []
        for trees in dataset_obj.nn_data:
            for sample in trees:
                sample_list.append(sample)
                sentiment_float = sample['root_node']['sentiment_label']
                sentiment_int = cfg.sentiment_float_to_int(sentiment_float, fine_grained)
                int_labels.append(sentiment_int)

        # check data
        assert len(sample_list) == pred_arr.shape[0]
        assert pred_arr.shape[0] == eval_arr.shape[0]

        # save csv
        with open(os.path.join(save_dir, 'sent_sample_res.csv'), 'w', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(['sent', 'label', 'pred', 'delta'])
            for sample, pred_val, int_label in zip(sample_list, pred_arr, int_labels):
                if sample['is_sent']:
                    sent = ' '.join(sample['root_node']['token_seq'])
                    label = int_label
                    pred = int(pred_val)
                    delta = int(math.fabs(label - pred))
                    csv_writer.writerow([sent, label, pred, delta])

        # statistics
        all_class_collect = []
        all_class_right_collect = []
        sent_class_collect = []
        sent_class_right_collect = []
        for sample, eval_val, int_label in zip(sample_list, eval_arr, int_labels):
            if sample['is_sent']:
                if float(eval_val) == 1.:
                    sent_class_right_collect.append(int_label)
                sent_class_collect.append(int_label)
            if float(eval_val) == 1.:
                all_class_right_collect.append(int_label)
            all_class_collect.append(int_label)
        all_class_pdf = nltk.FreqDist(all_class_collect)
        all_class_right_pdf = nltk.FreqDist(all_class_right_collect)
        sent_class_pdf = nltk.FreqDist(sent_class_collect)
        sent_class_right_pdf = nltk.FreqDist(sent_class_right_collect)

        with open(os.path.join(save_dir, 'statistics.txt'), 'w') as file:
            file.write('class ,all_class, all_class_right, all_rate, sent_class, sent_class_right, sent_rate' +
                       os.linesep)
            for i in range(out_class_num):
                all_class_num = all_class_pdf[i]
                all_class_right_num = all_class_right_pdf[i]
                sent_class_num = sent_class_pdf[i]
                sent_class_right_num = sent_class_right_pdf[i]
                file.write('%d, %d, %d, %.4f, %d, %d, %.4f'%
                           (i, all_class_num, all_class_right_num, 1.0*all_class_right_num/all_class_num,
                            sent_class_num, sent_class_right_num, 1.0*sent_class_right_num/sent_class_num))
                file.write(os.linesep)

        # statistics.csv
        all_class_table = [[] for _ in range(out_class_num)]
        sent_class_table = [[] for _ in range(out_class_num)]
        for sample, pred_val, int_label in zip(sample_list, pred_arr, int_labels):
            if sample['is_sent']:
                sent_class_table[int_label].append(pred_val)
            all_class_table[int_label].append(pred_val)
        all_class_table = [nltk.FreqDist(pred_list) for pred_list in all_class_table]
        sent_class_table = [nltk.FreqDist(pred_list) for pred_list in sent_class_table]
        with open(os.path.join(save_dir, 'statistics.csv'), 'w') as file:
            csv_writer = csv.writer(file)
            for i in range(out_class_num):
                row = [all_class_table[i][j] for j in range(out_class_num)]
                csv_writer.writerow(row)
            csv_writer.writerow([])
            for i in range(out_class_num):
                row = [sent_class_table[i][j] for j in range(out_class_num)]
                csv_writer.writerow(row)








class DatasetAnalysis(object):
    def __init__(self):

        pass

    @staticmethod
    def do_analysis(dataset_obj):
        # 1. all sample classification distribution
        # 2. all sentence sample classification distribution
        sample_num = dataset_obj.sample_num
        collect = []
        sent_collect = []
        for trees in dataset_obj.nn_data:
            for sample in trees:
                sentiment_float = sample['root_node']['sentiment_label']
                sentiment_int = cfg.sentiment_float_to_int(sentiment_float)
                if sample['is_sent']:
                    sent_collect.append(sentiment_int)
                collect.append(sentiment_int)
        all_pdf = nltk.FreqDist(collect)
        sent_pdf = nltk.FreqDist(sent_collect)
        print('sample_num:', sample_num)
        print('all')
        print(all_pdf.tabulate())
        print('sent')
        print(sent_pdf.tabulate())



if __name__ == '__main__':
    ifLoad, data = load_file(cfg.processed_path, 'processed data', 'pickle')
    assert ifLoad
    train_data_obj = data['train_data_obj']
    dev_data_obj = data['dev_data_obj']
    test_data_obj = data['test_data_obj']

    DatasetAnalysis.do_analysis(train_data_obj)







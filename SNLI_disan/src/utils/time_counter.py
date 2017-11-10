import time


class TimeCounter(object):
    def __init__(self):
        self.data_round = 0
        self.epoch_time_list = []
        self.batch_time_list = []

        # run time
        self.start_time = None

    def add_start(self):
        self.start_time = time.time()

    def add_stop(self):
        assert self.start_time is not None
        self.batch_time_list.append(time.time() - self.start_time)
        self.start_time = None

    def update_data_round(self, data_round):
        if self.data_round == data_round:
            return None, None
        else:
            this_epoch_time = sum(self.batch_time_list)
            self.epoch_time_list.append(this_epoch_time)
            self.batch_time_list = []
            self.data_round = data_round
            return this_epoch_time, \
                   1.0 * sum(self.epoch_time_list)/len(self.epoch_time_list) if len(self.epoch_time_list) > 0 else 0















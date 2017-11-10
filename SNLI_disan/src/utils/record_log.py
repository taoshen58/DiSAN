from configs import cfg
import os
import time


class RecordLog(object):
    def __init__(self,writeToFileInterval=20,  fileName = 'log.txt'):
        self.writeToFileInterval = writeToFileInterval
        self.waitNumToFile = self.writeToFileInterval
        buildTime = '-'.join(time.asctime(time.localtime(time.time())).strip().split(' ')[1:-1])
        buildTime = '-'.join(buildTime.split(':'))
        logFileName = buildTime#cfg.model_name[1:] + '_' + buildTime


        self.path = os.path.join(cfg.log_dir or cfg.standby_log_dir, logFileName+"_"+fileName)

        self.storage = []


    def add(self, content = '-'*30, ifTime = False , ifPrint = True , ifSave = True):
        #timeStr = "   ---" + str(time()) if ifTime else ''
        timeStr = "   --- "+time.asctime(time.localtime(time.time())) if ifTime else ''
        logContent = content + timeStr
        if ifPrint:
            print(logContent)
        #check save
        if ifSave:
            self.storage.append(logContent)
            self.waitNumToFile -= 1
            if self.waitNumToFile == 0:
                self.waitNumToFile = self.writeToFileInterval
                self.writeToFile()
                self.storage = []

    def writeToFile(self):

        with open(self.path, 'a', encoding='utf-8') as file:
            for ele in self.storage:
                file.write(ele + os.linesep)

    def done(self):
        self.add('Done')


    #def __del__(self):
    #    with open(self.path, 'a', encoding='utf-8') as file:
    #        for ele in self.storage:
    #            file.write(ele + os.linesep)

_logger = RecordLog(20)
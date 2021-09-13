import os
import datetime
import pickle

class Logger(object):
    def __init__(self, log_dir, subdir):
        self.log_dir = log_dir
        self.subdir = subdir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        local = os.path.join(log_dir, subdir)

        if not os.path.exists(local):
            os.makedirs(local)
        
        datenow =  datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
        self.curr_path = os.path.join(local, datenow)

        if  os.path.exists(self.curr_path):
            assert("about to overwrite important file")
        else:
            os.makedirs(self.curr_path)

        self.data = {}


    def log_scalar(self, tag, value):
        if tag not in self.data:
            self.data[tag] = []
        self.data[tag].append(value)
        
    def flush(self):
        for tag in self.data:
            with open(os.path.join(self.curr_path, tag+".pkl"), 'wb') as fp:
                print(len(self.data[tag]))
                pickle.dump(self.data[tag], fp)




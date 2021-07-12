from collections import defaultdict
import os
from preprocessing.params import create_dir
from train.il_trainer import IL_Trainer
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime 



class Recorder(object):
    def __init__(self, il_trainer: IL_Trainer):
        self.il_trainer = il_trainer
        self.writer = None

        self.cur_state = il_trainer.cur_state
        self.iteration = 0
        self.losses = defaultdict(list)

        # create runs dir
        self.root_path = os.path.join(self.il_trainer.params['root_dir'], 'runs')
        create_dir(self.root_path)

        self.enable_record = self.il_trainer.params['record']
        self.init_writer()
        

    def init_writer(self):
        if not self.enable_record:
            return
        if self.writer != None:
            self.writer.close()
            del self.writer
        now = datetime.now().strftime("%Y-%m-%d-%H-%M")

        logdir = os.path.join(self.root_path, now + '_' +self.il_trainer.params['scenario'])

        self.writer = SummaryWriter(logdir)
    
    def next_state(self):
        if not self.enable_record:
            return
        self.cur_state += 1
        self.iteration = 0
    
    def add_iter_loss(self, losses:dict):
        if not self.enable_record:
            return
        for key, value in losses.items():
            tag = 'Train_iter_loss/state{}/{}'.format(self.cur_state,key)
            self.writer.add_scalar(tag=tag,
                                    scalar_value=value, 
                                    global_step=self.iteration)
            self.losses[key].append(value)
        self.iteration += 1
    def record_epoch_loss(self, epoch:int):
        if not self.enable_record:
            return
        for key, value in self.losses.items():
            tag = 'Train_epoch_loss/state{}/{}'.format(self.cur_state,key)
            self.writer.add_scalar(tag=tag,
                                    scalar_value=np.mean(value), 
                                    global_step=epoch - 1)

        self.losses = defaultdict(list)

    def end_write(self):
        if not self.enable_record:
            return
        self.writer.close()

    
        
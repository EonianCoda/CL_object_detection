# built-in
import collections
import os
import pickle
# torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
# retinanet
from retinanet.dataloader import AspectRatioBasedSampler, IL_dataset, Replay_dataset, Resizer, Augmenter, Normalizer, collater
from retinanet.model import create_retinanet
# traing util
from preprocessing.params import Params
from preprocessing.debug import debug_print, DEBUG_FLAG
# IL
from IL_method.mas import MAS
from IL_method.agem import A_GEM


class IL_Trainer(object):
    def __init__(self, params:Params, model, optimizer, scheduler, dataset_train:IL_dataset, loss_hist=None):
        self.params = params
 
        # training setting
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset_train = dataset_train
        if loss_hist == None:
            self.loss_hist = collections.deque(maxlen=500)
        else:
            self.loss_hist = loss_hist
        self.dataloader_train = None
        self.update_dataloader()
            
        self.cur_state = self.params['start_state']
        
        # incremental tools
        self.prev_model = None
        self.dataset_replay = None
        self.mas = None
        self.agem = None

        # when training, use above attribute
        self.cur_warm_stage = -1

        # if start state is not initial state, then update incremental learning setting
        if self.cur_state >= 1:
            self.init_agem()
            self.init_replay_dataset()
            self.update_prev_model()
            self.update_mas()

    def update_dataloader(self):
        if self.dataloader_train != None:
            del self.dataloader_train
        sampler = AspectRatioBasedSampler(self.dataset_train, batch_size = self.params['batch_size'], drop_last=False)
        self.dataloader_train = DataLoader(self.dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)

    def update_prev_model(self):
        """update previous model, if distill = True
        """
        if self.cur_state == 0:
            raise ValueError("Initial state doesn't have previous state")
        if not self.params['distill']:
            return

        if self.prev_model != None:
            self.prev_model.cpu()
            del self.prev_model
        self.prev_model = create_retinanet(self.params['depth'], num_classes=self.params.states[self.cur_state - 1]['num_knowing_class'])
        self.params.load_model(self.cur_state - 1, -1, self.prev_model)
        self.prev_model.training = False
        self.prev_model.cuda()

    def init_replay_dataset(self):
        # Replay dataloader
        if self.params['sample_num'] <= 0:
            return
        
        self.dataset_replay = Replay_dataset(self.params,
                                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        custom_ids = []
        # custom sample
        if self.params['sample_method'] == 'herd':
            with open(os.join(self.params['ckp_path'], 'state{}'.format(self.cur_state - 1), 'examplar.pickle'), 'rb') as f:
                examplar = pickle.load(f)
            examplar = ['{:06d}'.format(img_id) for img_id in examplar]
            self.dataset_replay.reset_by_imgIds(per_num=self.params['sample_num'], img_ids=examplar)
        else:
            self.dataset_replay.reset_by_state(self.cur_state)
        sampler = AspectRatioBasedSampler(self.dataset_replay, batch_size = self.params['batch_size'], drop_last=False)
        self.dataloader_replay = DataLoader(self.dataset_replay, num_workers=2, collate_fn=collater, batch_sampler=sampler)
 
    def init_agem(self):
        if not self.params['agem']:
            self.agem = None
            return
        self.agem = A_GEM(self.model, self.dataset_replay, self.params)

    def update_mas(self):
        # set MAS penalty
        if not self.params['mas']:
            return

        debug_print("Update MAS")
        if self.mas != None:
            del self.mas
        self.mas = MAS(self.model, self.params)
        # Test if the mas file exists
        mas_file = os.path.join(self.params['ckp_path'], "state{}".format(self.cur_state - 1), "{}.pickle".format(self.params['mas_file']))
        if not self.mas.load_importance(mas_file):
            self.mas.calculate_importance(self.dataloader_train)

    def update_training_tools(self):
        self.model.next_state(self.get_cur_state()['num_new_class'])
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

    def next_state(self):
        self.cur_state += 1
        self.update_mas()
        self.update_training_tools()
        self.dataset_train.next_state()
        
        if self.dataset_replay !=None:
            if self.cur_state == 1:
                self.init_replay_dataset()
            else:
                self.dataset_replay.next_state()
        if self.cur_state == 1:
            self.init_agem()
        
        self.update_dataloader()
        self.update_prev_model(self.cur_state - 1)

    def warm_up(self, epoch:int):
        # No warm-up
        if self.params['warm_stage'] == 0:
            self.warm_status = 0
            self.cur_warm_stage = -1
            return

        cur_warm_stage , white_list = self.params.is_warmup(epoch)
        if white_list != None:
            self.model.freeze_layers(white_list)
        else:
            self.model.unfreeze_layers()
        self.cur_warm_stage = cur_warm_stage

    def save_ckp(self, epoch_loss:list,epoch:int):

        self.params.save_checkpoint(self.cur_state, epoch, self.model, self.optimizer, self.scheduler, self.loss_hist, epoch_loss)
    def get_cur_state(self):
        return self.params.states[self.cur_state]

    def destroy(self):
        if self.model != None:
            self.model.cpu()
            del self.model
        if self.optimizer != None:
            self.optimizer.cpu()
            del self.optimizer
        if self.mas != None:
            self.mas.destroy()
        self.params = None

        

# built-in

from IL_method.persuado_label import Labeler
from IL_method.prototype import ProtoTyper
from IL_method.bic import Bic_Trainer
import collections
import os
import pickle
import matplotlib.pyplot as plt
import cv2
# torch
import torch
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
from IL_method.herd_sample import Herd_sampler
from IL_method.weight_init import get_similarity


WHITE_LIST_FOR_OPTIM = ['classificationModel.output']

def get_parameters(model, white_list=[]):
    def check(target):
        for name in white_list:
            if name in target:
                return False
        return True
    
    for name, p in model.named_parameters():
        if not check(name):
            # print(name)
            continue
        else:
            yield p

class IL_Trainer(object):
    def __init__(self, params:Params, model, optimizer, scheduler, dataset_train:IL_dataset, loss_hist=None):
        self.params = params
        self.cur_epoch = 0 

        
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
        self.dataloader_replay = None
        self.mas = None
        self.agem = None
        self.bic = None
        

        # when training, use above attribute
        self.cur_warm_stage = -1

        # if start state is not initial state, then update incremental learning setting
        if self.cur_state >= 1:
            # The order below is important, because some method dependent on another component
            self.init_prototyper()
            self.init_replay_dataset()
            self.init_bic()
            self.update_replay_dataloader()
            self.init_agem()
            self.update_prev_model()
            self.update_mas()

            self.add_persuado_label()

    def add_persuado_label(self):
        if self.params['persuado_label'] == False:
            return
        labler = Labeler(self.model, self.params)
        persuado_label = labler.get_persuado_label(self.cur_state)
        self.dataset_train.persuado_label = persuado_label
        self.update_dataloader()

    def init_bic(self):
        if not self.params['bic']:
            return
        if self.dataset_replay == None:
            raise(ValueError("Please call init_replay_dataset first"))
        self.bic = Bic_Trainer(self, self.params['bic_ratio'])

        if self.params['start_epoch'] != 1:
            path = os.path.join(self.params['ckp_path'],'state{}'.format(self.cur_state), 'bic_{}.pt'.format(self.params['start_epoch']))
            self.bic.load_ckp(path)
        self.update_dataloader()
        self.update_replay_dataloader()
            
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
        if not self.params['distill'] and not self.params['mas']:
            return

        if self.prev_model != None:
            self.prev_model.cpu()
            del self.prev_model
        self.prev_model = create_retinanet(self.params['depth'], num_classes=self.params.states[self.cur_state - 1]['num_knowing_class'])
        self.params.load_model(self.cur_state - 1, -1, self.prev_model)
        self.prev_model.training = False
        self.prev_model.cuda()

    def init_prototyper(self):
        if self.params['prototype_loss'] or self.params['sample_method'] == 'prototype_herd':
            self.protoTyper = ProtoTyper(self)
            if self.params['sample_method'] == 'prototype_herd':
                # path = os.path.join(self.params['ckp_path'], 'state{}'.format(self.cur_state - 1))
                # file_name = 'classification_herd_samples.pickle'
                # if not os.path.isfile(os.path.join(path, file_name)):
                self.protoTyper.cal_examplar(self.cur_state - 1)
            if not self.params['prototype_loss']:
                del self.protoTyper
            elif self.protoTyper.prototype_features == None:
                self.protoTyper.init_prototype(self.cur_state - 1)
        

    def init_replay_dataset(self):
        """init reaply dataset if params['sample_num'] > 0
        """
        # Replay dataloader
        if self.params['sample_num'] <= 0:
            return
        
        self.dataset_replay = Replay_dataset(self.params,
                                            transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        # if self.params['sample_method'] == 'herd':
        if self.params['sample_method'] == 'herd':
            
            self.herd_sampler = Herd_sampler(self)
            self.herd_sampler.sample(self.params['sample_num'])
            self.dataset_replay.reset_by_imgIds(per_num=self.params['sample_num'], img_ids=self.herd_sampler.examplar_list)

        elif self.params['sample_method'] == 'prototype_herd':
            path = os.path.join(self.params['ckp_path'], 'state{}'.format(self.cur_state - 1))
            file_name = 'classification_herd_samples.pickle'
            if not os.path.isfile(os.path.join(path, file_name)):
                raise ValueError("Unkowing Error in init prototype_herd")
            with open(os.path.join(path, file_name), 'rb') as f:
                sample_dict, count = pickle.load(f)

            coco = self.params.states.coco


            knowing_class_ids = self.params.states[self.cur_state - 1]['knowing_class']['id']
            future_ids = []
            for i in range(1, 20 + 1):
                if i not in knowing_class_ids:
                    future_ids.append(i)

            future_img_ids = coco.get_imgs_by_cats(future_ids)


            count = count.squeeze()
            ranked_count = torch.argsort(count, descending=True).int()

            examplar_dict = {}
            sample_img_ids = []
            per_num = self.params['sample_num']
            
            num_anchors = len(ranked_count[0])
            sample_for_each_anchor = [0 for _ in range(num_anchors)]
            
            i = 0
            for _ in range(per_num):
                sample_for_each_anchor[i] += 1
                i = (i + 1) % num_anchors


            for class_id in sample_dict.keys():
                examplar_dict[class_id] = []
                for idx, anchor_id in enumerate(ranked_count[class_id]):
                    anchor_id = int(anchor_id)

                    cur_anchor_num_sample = sample_for_each_anchor[idx]
                    if cur_anchor_num_sample == 0:
                        continue
                    for img_id in sample_dict[class_id][anchor_id]:
                        if img_id not in sample_img_ids and img_id not in future_img_ids:
                            sample_img_ids.append(img_id)
                            examplar_dict[class_id].append(img_id)
                            cur_anchor_num_sample -= 1
                            if cur_anchor_num_sample == 0:
                                break
  
            self.dataset_replay.reset_by_imgIds(per_num=self.params['sample_num'], img_ids=sample_img_ids)

        # elif self.params['sample_method'] == 'maxNum':
        #     path = os.path.join(self.params['ckp_path'], 'state{}'.format(self.cur_state - 1))
        #     with open(os.path.join(path, 'maxNum_scores.pickle'), 'rb') as f:
        #         classed_max_lists = pickle.load(f)
        #     examplar = []
        #     for cat_id in self.params.states[self.cur_state - 1]['knowing_class']['id']:
        #         nums = [num for num in classed_max_lists[cat_id].keys()]
        #         nums.sort(reverse=True)
        #         cur_num = 0
        #         for num in nums:
        #             for img_id in classed_max_lists[cat_id][num]:
        #                 if img_id not in examplar:
        #                     examplar.append(img_id)
        #                     cur_num += 1
        #                     if cur_num == self.params['sample_num']:
        #                         break
        #             if cur_num == self.params['sample_num']:
        #                 break
        #     self.dataset_replay.reset_by_imgIds(self.params['sample_num'], examplar)
        else: # random sample
            self.dataset_replay.reset_by_state(self.cur_state)

        # save samples png
        img_path = self.dataset_train.image_path
        replay_imgs = self.dataset_replay.image_ids
        num_classes = int(len(replay_imgs) / self.params['sample_num'])
        
        path = os.path.join(self.params['ckp_path'], 'state{}'.format(self.cur_state))

        with open(os.path.join(path,'examplar.txt'), 'w') as f:
            for img_id in replay_imgs:
                f.write("{}\n".format(img_id))

        cat_ids = self.params.states[-1]['knowing_class']['id'][:num_classes]
        cat_names = self.params.states.coco.catId_to_name(cat_ids)

        #TODO 當為herd時，應改變順序
        if self.params['output_examplar']:
            fig = plt.figure(figsize=(4*self.params['sample_num'],3.5*num_classes), constrained_layout=True)
            gs = fig.add_gridspec(num_classes, self.params['sample_num'])
            row = 0
            for row, cat_name in enumerate(cat_names):
                for col in range(self.params['sample_num']):
                    ax = fig.add_subplot(gs[row, col])
                    im = cv2.imread(os.path.join(img_path, "{:06d}".format(replay_imgs[row*self.params['sample_num']+col])) +'.jpg')
                    ax.set_title(cat_name)
                    ax.imshow(im)
            
            file_name = os.path.join(path, "examplar.png")
            plt.savefig(file_name)

    def update_replay_dataloader(self):
        if self.params['sample_num'] <= 0:
            return
        if self.dataloader_replay != None:
            del self.dataloader_replay
        sampler = AspectRatioBasedSampler(self.dataset_replay, batch_size = self.params['sample_batch_size'], drop_last=False)
        self.dataloader_replay = DataLoader(self.dataset_replay, num_workers=2, collate_fn=collater, batch_sampler=sampler)
 
    def init_agem(self):
        if not self.params['agem']:
            self.agem = None
            return

        num_groups = int((len(self.dataset_replay.image_ids) - 1) / self.params['batch_size']) + 1
        self.agem = A_GEM(self.dataloader_replay, num_groups)

    def update_mas(self):
        # set MAS penalty
        if not self.params['mas']:
            return

        debug_print("Update MAS")
        if self.mas != None:
            self.mas.destory()
            del self.mas
        self.mas = MAS(self.model, self.params)
        # Test if the mas file exists
        if not self.mas.load_importance(state=self.cur_state - 1):
            self.mas.calculate_importance(self.dataloader_train,self.cur_state - 1)

    def update_training_tools(self):
        """update model, optimizer and scheduler
        """

        method = self.params['init_method']
        if  method == "large" or method == "mean":
            debug_print("{} Similarity init ".format(self.params['init_method']))
            similarity_file = os.path.join(self.params['ckp_path'], "state{}".format(self.cur_state - 1), "similarity.pickle")
            if os.path.isfile(similarity_file):
                with open(similarity_file, 'rb') as f:
                    similaritys = pickle.load(f) 
            else:
                similaritys = get_similarity(self.model, self.dataset_train)
                with open(similarity_file, 'wb') as f:
                    pickle.dump(similaritys, f)
        else:
            similaritys = None

        self.model.next_state(self.get_cur_state()['num_new_class'], similaritys, method)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.params['lr'])
        self.optimizer = optim.Adam([{'params':get_parameters(self.model, WHITE_LIST_FOR_OPTIM)},
                                    {'params':self.model.classificationModel.output.parameters()}]
                                    , lr=self.params['lr'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.params['scheduler_milestone'], gamma=self.params['scheduler_decay'], verbose=True)

        #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=True)

    def next_state(self):
        self.cur_state += 1
        self.update_mas()
        self.dataset_train.next_state()
        self.update_training_tools()
        
    
        if self.dataset_replay !=None:
            if self.cur_state == 1:
                self.init_replay_dataset()
                self.init_bic()
                self.init_agem()

            else:
                self.dataset_replay.next_state()  
            self.update_replay_dataloader()
 
        self.update_dataloader()
        self.update_prev_model()

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
        if self.params['bic'] and self.cur_state > 0:
            bic_ckp_path = os.path.join(self.params['ckp_path'],'state{}'.format(self.cur_state), 'bic_{}.pt'.format(epoch))
            self.bic.save_ckp(bic_ckp_path)
    
    def get_cur_state(self):
        return self.params.states[self.cur_state]

    def auto_delete(self, state:int, epoch:int):
        self.params.auto_delete(state, epoch)
        if self.params['bic'] and self.cur_state > 0:
            for i in range(1, epoch):
                if i % 5 == 0:
                    continue
                path = os.path.join(self.params['ckp_path'], 'state{}'.format(self.cur_state),'bic_{}.pt'.format(i))
                if os.path.isfile(path):
                    os.remove(path)

    def destroy(self):
        if self.model != None:
            self.model.cpu()
            del self.model
        if self.optimizer != None:
            del self.optimizer  
        if self.mas != None:
            self.mas.destroy()
        self.params = None

        

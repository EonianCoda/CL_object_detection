# import collections

import pickle
from posix import EX_OSFILE
from preprocessing.debug import debug_print, DEBUG_FLAG
from preprocessing.enhance_coco import Enhance_COCO
from retinanet.model import create_retinanet

import os
import torch
import random
import copy
WARM_UP_WHITE_LIST = { 'output':['classificationModel.output'],
                        'fpn':['classificationModel', 'regressionModel'],
                       'resnet':['fpn', 'classificationModel', 'regressionModel']
                       }

# num_knowing_class = num_new_class + num_past_class
EMPTY_STATE = {'knowing_class':{'id':[],'name':[]},
                'new_class':{'id':[],'name':[]},
                'num_past_class':0, 
                'num_new_class':0, 
                'num_knowing_class':0}


def create_dir(path):
    """check whether directory exists or not. If not, then create it 
    """
    if not os.path.isdir(path):
        os.mkdir(path)


class IL_states(object):
    def __init__(self, coco_path: str, scenario_list:list, shuffle_class:bool):
        
        self.coco = Enhance_COCO(coco_path)
        self.total_states_num = len(scenario_list)
        self.init_states(scenario_list, shuffle_class)


    def init_states(self, scenario_list:list, shuffle = False):
        """init incremeantal learning task
            Args:
                scenario: the incremental learning scenario
                shuffle: whether shuffle the order
        """

        self.states = dict([(idx, copy.deepcopy(EMPTY_STATE)) for idx, _ in enumerate(scenario_list)])

        classes = sorted(self.coco.classes.values())
        if shuffle:
            random.shuffle(classes)

        # for 15+1=train
        if scenario_list == [15,1]:
            classes[15] = 'train'


        total_num = 0
        for idx, num in enumerate(scenario_list):
            self.states[idx]['num_new_class'] = num
            total_num += num
            # non-incremental initial state
            if idx == 0:
                self.states[idx]['new_class']['name'].extend(classes[:total_num])
                self.states[idx]['new_class']['id'] = self.coco.catName_to_id(self.states[idx]['new_class']['name'], sort=True)
                self.states[idx]['knowing_class']['name'] = self.states[idx]['new_class']['name']
                self.states[idx]['knowing_class']['id'] = self.states[idx]['new_class']['id']
                self.states[idx]['num_knowing_class'] = num
                continue

            # incremental state
            self.states[idx]['num_past_class'] = self.states[idx - 1]['num_knowing_class']
            self.states[idx]['num_knowing_class'] = total_num

            self.states[idx]['new_class']['name'].extend(classes[total_num - num:total_num])
            # it is well-sorted
            self.states[idx]['new_class']['id'] = self.coco.catName_to_id(self.states[idx]['new_class']['name'], sort=True)


            self.states[idx]['knowing_class']['name'].extend(self.states[idx - 1]['knowing_class']['name'])
            self.states[idx]['knowing_class']['name'].extend(self.states[idx]['new_class']['name'])
            self.states[idx]['knowing_class']['id'].extend(self.states[idx - 1]['knowing_class']['id'])
            self.states[idx]['knowing_class']['id'].extend(self.states[idx]['new_class']['id'])

        self.total_class_num = total_num

    def __getitem__(self, key):
        if isinstance(key, int) and key < 0:
            key = list(self.states.keys())[key]
        return self.states[key]

    def __len__(self):
        return self.total_states_num

    def print_state_info(self):
        print("Total State number = {}".format(self.total_states_num))
        print("Total Class number = {}".format(self.total_class_num))
        for idx, state in enumerate(self.states.values()):
            print("State {}:".format(idx))
            print("New class number = {}".format(state['num_new_class']))
            print("Knowing class number = {}".format(state['num_knowing_class']))
            print("New class:")
            print("\t Name = ", state['new_class']['name'])
            print("\t Id = ", self.coco.catName_to_id(state['new_class']['name']))

class Params(object):
    def __init__(self, parser:dict, specific_data_split = None):
        """ Parse information from parser, and provide some helpful function
            Args:
                parser: dict
                specific_data_split: for validation , default = None
        """

        self._params = parser

        self['scenario_list'] = parser['scenario']
        self['scenario'] = "_".join([str(i) for i in parser['scenario']])

        # init datasplit
        if specific_data_split == None:
            if self['dataset'] == "voc2007":
                self['data_split'] = "trainval"
            else: # for voc2012
                self['data_split'] = "train"
        else:
            self['data_split'] = specific_data_split

        # init path and directory
        ckp_path = os.path.join(self['root_dir'], 'checkpoint')
        create_dir(ckp_path)
        ckp_path = os.path.join(ckp_path, self['scenario'])
        create_dir(ckp_path)
        self['ckp_path'] = ckp_path #  checkpoint path, no state num, for example "root_dir/checkpoint/15_1", "root_dir/checkpoint/20"
        self['data_path'] = os.path.join(self['root_dir'], 'dataset', self['dataset']) # the training data path

        # init states for scenario
        coco_path = os.path.join(self['data_path'], 'annotations', '{}_{}.json'.format(self['dataset'], self['data_split']))
        self.states = IL_states(coco_path, self['scenario_list'], self['shuffle_class'])

        # init warmup setting
        self.init_warmup()

    def __setitem__(self, key, value):
        self._params[key] = value
    def __getitem__(self, key):
        return self._params[key]

    def init_warmup(self):
        if self['warm_stage'] == 0:
            return
        # Exception
        if len(self['warm_epoch']) != self['warm_stage']:
            raise ValueError("The number of warm stage must match the warm epochs")

        self['warm_stop_epoch'] = [self['warm_epoch'][0] + 1]
        for i in range(1, len(self['warm_epoch'])):
            self['warm_stop_epoch'].append(self['warm_stop_epoch'][-1] + self['warm_epoch'][i])
        
        self['warm_white_list'] = []
        for key in self['warm_layers']:
            self['warm_white_list'].append(WARM_UP_WHITE_LIST[key])

    def is_warmup(self, epoch:int):
        """check whether it is warm up epoch or not
            Args:
                epcoh: current epoch
            Return:
                success: (currrent warmup stage, white_list)
                fail: (-1, None)
        """
        if self['warm_stage'] == 0:
            return (-1, None)
        
        for idx, stop_epoch in enumerate(self['warm_stop_epoch']):
            if epoch < stop_epoch:
                return (idx, self['warm_white_list'][idx])
        return (-1, None)

    def auto_delete(self, state : int, epoch : int):
        """delete the checkpoint every five checkpoint

            Args:
                state: current state index 
                epoch: current epoch num
        """
        for i in range(1, epoch):
            if i % 5 == 0:
                continue
            if os.path.isfile(self.get_ckp_path(state, i)):
                os.remove(self.get_ckp_path(state, i))

    def get_ckp_path(self, state:int, epoch:int):
        """get checkpoint's path, which contains the checkpoint's name, for example: "/checkpoint/20/voc2007_checkpoint_10.pt"

            Args:
                state: current state index 
                epoch: current epoch num
        """
        ckp_path = os.path.join(self['ckp_path'], "state{}".format(state))
        ckp_name = '{}_checkpoint_{}.pt'.format(self['dataset'], epoch)
        create_dir(ckp_path)
        return os.path.join(ckp_path, ckp_name)

    def load_checkpoint(self, state:int, epoch:int):
        """read checkpoint

        Args:
            state: state index
            epoch: a torch epoch , if epoch = -1, then auto search the newest model in current state
        Return: 
            checkpoint
        """
        if epoch == -1:
            ckp_path = os.path.join(self['ckp_path'], "state{}".format(state))
            ckp_names = [ckp_name for ckp_name in os.listdir(ckp_path) if '.pt' in ckp_name]
            epoch = max([int(name.split('_')[-1].split('.')[0]) for name in ckp_names])

        debug_print('Load checkpoint at state{} Epoch{}'.format(state, epoch)) 
        return torch.load(self.get_ckp_path(state, epoch))
       
    def get_model(self, state:int,  epoch=-1):
        """read checkpoint
            Args:
                state: state index
                epoch: prefered epochs, -1 mean auto search the max epoch in this state , None mean not readcheckpoint default = None 
        """
        if state < 0:
            raise ValueError("State{} doesn't exist".format(state))

        model = create_retinanet(self['depth'], self.states[state]['num_knowing_class'])
        self.load_model(state, epoch, model)
        return model

    def load_model(self, state:int, epoch:int, model = None, optimizer = None, scheduler = None, loss_hist = None):
        """read checkpoint

            Args:
                state: state index
                epoch: prefered epochs
                model: a torch model
                optimizer: a torch optimizer
                scheduler: a torch scheduler
        """
        
        if state < 0:
            raise ValueError("State{} doesn't exist".format(state))
        ckp = self.load_checkpoint(state, epoch)
        if model != None:
            model.load_state_dict(ckp['model_state_dict'])
        if optimizer != None:
            optimizer.load_state_dict(ckp['optimizer_state_dict'])
        if scheduler != None:
            scheduler.load_state_dict(ckp['scheduler_state_dict'])
        if scheduler != None:
            # compatible to old checkpoint
            if ckp.get('loss_hist'):
                loss_hist = ckp['loss_hist']

    def save_checkpoint(self, state:int, epoch:int, model, optimizer=None, scheduler=None, loss_hist=None, epoch_loss=None):
        """ save checkpoint
        """
        debug_print('Save checkpoint at state{} Epoch{}'.format(state, epoch))
        save_path = self.get_ckp_path(state, epoch)
        data = {'epoch':epoch, 'model_state_dict': model.state_dict()}
        if optimizer != None:
            data['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler != None:
            data['scheduler_state_dict'] = scheduler.state_dict()
        if loss_hist != None:
            data['loss_hist'] = loss_hist
        if epoch_loss != None:
            data['epoch_loss'] = epoch_loss
            
        torch.save(data, save_path)


    def _il_keyword_check(self, name):
        keyword = ['distill', 
            'sample', 
            'mas', 
            'warm',
            'enhance',
            'ignore',
            'decrease_positive',
            'method']
  
        for word in keyword:
            if word in name:
                return True
        return False

    def get_il_info(self):
        def to_str(value):
            if isinstance(value, list):
                return ",".join([str(v) for v in value])
            elif isinstance(value, bool):
                if value:
                    return "True"
                else:
                    return "False"
            else:
                return value
        result = dict()

        # warm-up
        result['warm_stage'] = to_str(self['warm_stage'])
        if self['warm_stage'] == 0:
            result['warm_epoch'] = "None"
            result['warm_layers'] = "None"
        else:
            result['warm_epoch'] = to_str(self['warm_epoch']) 
            result['warm_layers'] = to_str(self['warm_layers']) 
        # distill
        result['distill'] = to_str(self['distill'])
        result['distill_logits'] = to_str(self['distill_logits'])
        if self['distill_logits']:
            result['distill_logits_on'] = to_str(self['distill_logits_on'])
            result['distill_logits_bg_loss'] = to_str(self['distill_logits_bg_loss'])
        else:
            result['distill_logits_on'] = "None"
            result['distill_logits_bg_loss'] = "None"

        # Sample
        result['sample_num'] = to_str(self['sample_num'])
        if self['sample_num'] > 0:
            result['sample_method'] = to_str(self['sample_method'])
        else:
            result['sample_method'] = "None"
        
        # Mas
        result['mas'] = to_str(self['mas'])
        # A-gem
        result['agem'] = to_str(self['agem'])


        # Experimental params
        result['ignore_ground_truth'] = to_str(self['ignore_ground_truth'])
        result['decrease_positive'] = to_str(self['decrease_positive'])

        result['enhance_error'] = to_str(self['enhance_error'])
        if self['enhance_error']:
            result['enhance_error_method'] = to_str(self['enhance_error_method'])
        else:
            result['enhance_error_method'] = "None"

        result['ignore_ground_truth'] = to_str(self['ignore_ground_truth'])
        result['init_method'] = to_str(self['init_method'])
        result['ignore_past_class'] = to_str(self['ignore_past_class'])
          

        return result
    def print_il_info(self):
        for key, value in self._params.items():
            if self._il_keyword_check(key):
                if isinstance(value, str):
                    print('{} = "{}"'.format(key, value))
                else:
                    print('{} = {}'.format(key, value))

    def output_params(self, state):

        output_path = os.path.join(self['ckp_path'], "state{}".format(state))

        # output il_hparams for validation
        with open(os.path.join(output_path, "il_hparams.pickle"), 'wb') as f:
            pickle.dump(self.get_il_info(), f)

        with open(os.path.join(output_path, "params.txt"), 'w') as f:
            lines = []
            
            for key, value in self._params.items():
                if isinstance(value, str):
                    lines.append('{} = "{}"'.format(key, value))
                else:
                    lines.append('{} = {}'.format(key, value))

            # states information
            lines.append("-"*100)

            lines.append("Total State number = {}".format(self.states.total_states_num))
            lines.append("Total Class number = {}".format(self.states.total_class_num))
            for idx, state in enumerate(self.states.states.values()):
                lines.append("State {}:".format(idx))
                lines.append("\tNew class number = {}".format(state['num_new_class']))
                lines.append("\tKnowing class number = {}".format(state['num_knowing_class']))
                lines.append("\tNew class:")
                lines.append("\t\tName = {}".format(state['new_class']['name']))

                ids = self.states.coco.catName_to_id(state['new_class']['name'], sort=False)
                lines.append("\t\tId = {}".format(ids))
                ids.sort()
                lines.append("\t\tSorted_Id = {}".format(ids))
            
            f.write('\n'.join(lines) + '\n')
                
            


        


            

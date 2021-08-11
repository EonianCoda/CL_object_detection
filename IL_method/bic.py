import torch
import torch.nn as nn
import torch.optim
from torch.optim.lr_scheduler import StepLR
import copy
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from retinanet.dataloader import AspectRatioBasedSampler, Bic_dataset, Resizer, Augmenter, Normalizer, collater
from retinanet.losses import IL_Loss

class BiasLayer(nn.Module):
    def __init__(self):
        super(BiasLayer, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, requires_grad=True, device="cuda"))
        self.beta = nn.Parameter(torch.zeros(1, requires_grad=True, device="cuda"))
    def forward(self, x):
        return self.alpha * x + self.beta
    def printParam(self, i):
        print(i, self.alpha.item(), self.beta.item())

class Bic_Evaluator(object):
    def __init__(self, params):
        self.params = params
        num_state = len(self.params.states) 
        self.bias_layers = [BiasLayer().cuda() for _ in range(num_state - 1)]
        self.num_init_class = self.params.states[0]['num_new_class']
        self.num_new_class = []
        for i in range(1, len(self.params.states)):
            self.num_new_class.append(self.params.states[i]['num_new_class'])

    def load_ckp(self, path:str):
        """load the checkpoint for bic layers model and scheduler
        """
        ckp = torch.load(path)
        for i in range(len(self.bias_layers)):
            self.bias_layers[i].load_state_dict(ckp['model_state_dict'][i])

    def bic_correction(self, x):
        """do bic correction
        Args:
            x: a tensor for the result of classification
        """
        count = self.num_init_class

        x_splits = []
        for i in range(self.cur_state):
            x_splits.append(x[:,:, count:count + self.num_new_class[i]])
            count += self.num_new_class[i]

        out = [x[:,:,:self.num_init_class]]
        for i in range(len(x_splits)):
            out.append(self.bias_layers[i](x_splits[i]))

        return torch.cat(out, dim=2)
    
# TODO 目前只能支援多一個新state
class Bic_Trainer(object):
    def __init__(self, il_trainer, val_ratio=0.1):
        self.il_trainer = il_trainer
        self.cur_state = il_trainer.cur_state

        self.per_num = max(int(self.il_trainer.params['sample_num'] * val_ratio), 1)

        self.il_loss = IL_Loss(self.il_trainer)
        self.optim = None
        # init bias_layer
        num_state = len(self.il_trainer.params.states)     
        self.bias_layers = [BiasLayer().cuda() for _ in range(num_state - 1)]
        self.num_init_class = self.il_trainer.params.states[0]['num_new_class']
        self.num_new_class = []
        for i in range(1, len(self.il_trainer.params.states)):
            self.num_new_class.append(self.il_trainer.params.states[i]['num_new_class'])

        self._sample_img()
        self._init_dataset()
        self.freeze()
        self.update_tools()

    def _init_dataset(self):
        self.image_ids
        self.dataset_bic = Bic_dataset(self.il_trainer.params, 
                                        transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
                                        self.image_ids,
                                        self.seen_ids)

        # sampler = AspectRatioBasedSampler(self.dataset_bic, batch_size = self.il_trainer.params['batch_size'], drop_last=False)
        sampler = AspectRatioBasedSampler(self.dataset_bic, batch_size = 4, drop_last=False)
        self.dataloader_bic = DataLoader(self.dataset_bic, num_workers=2, collate_fn=collater, batch_sampler=sampler)

    def update_tools(self):
        if self.optim != None:
            del self.optim

        self.optim = torch.optim.Adam(self.bias_layers[self.cur_state - 1].parameters(), lr=0.001)

    def _sample_img(self):
        """sample data for validation in bic method
        """
        self.seen_ids = []
        self.image_ids = []
        new_data = copy.deepcopy(self.il_trainer.dataset_train.image_ids)
        old_data = copy.deepcopy(self.il_trainer.dataset_replay.image_ids)

        # sample old data
            
        seen_ids = self.il_trainer.dataset_replay.seen_class_id
        for start_idx in range(0,len(old_data), self.il_trainer.params['sample_num']):
            for i in range(self.per_num):
                img_id = old_data[start_idx + i]
                self.image_ids.append(img_id)
                self.seen_ids.append(seen_ids)
                self.il_trainer.dataset_replay.image_ids.remove(img_id)


        # sample on new data
        states = self.il_trainer.params.states
        coco = states.coco
        seen_ids = self.il_trainer.dataset_train.seen_class_id
        new_class_ids = states[self.cur_state]['new_class']['id']
        for cat_id in new_class_ids:
            img_ids = coco.get_imgs_by_cats(cat_id)
            new_class_img_ids = list(set(img_ids) & set(new_data))
            new_class_img_ids.sort()

            for i in range(self.per_num):
                img_id = new_class_img_ids[i]
                self.image_ids.append(img_id)
                self.seen_ids.append(seen_ids)
                self.il_trainer.dataset_train.image_ids.remove(img_id)
                new_data.remove(img_id)

    def save_ckp(self, path:str):
        """save the checkpoint for bic layers
        """

        data = {'optim_state_dict':self.optim.state_dict(), 
                'model_state_dict': [bias_layer.state_dict() for bias_layer in self.bias_layers]}
        torch.save(data, path)

    def load_ckp(self, path:str):
        """load the checkpoint for bic layers model and scheduler
        """
        ckp = torch.load(path)
        for i in range(len(self.bias_layers)):
            self.bias_layers[i].load_state_dict(ckp['model_state_dict'][i])

        self.optim.load_state_dict(ckp['optim_state_dict'])
        self.freeze()

    def freeze(self):
        """freeze the bic layers
        """
        for i in range(len(self.bias_layers)):
            self.bias_layers[i].eval()
            self.bias_layers[i].alpha.requires_grad = False
            self.bias_layers[i].beta.requires_grad = False

    def unfreeze(self):
        """unfreeze the bic layers
        """
        for i in range(len(self.bias_layers)):
            self.bias_layers[i].train()
            self.bias_layers[i].alpha.requires_grad = True
            self.bias_layers[i].beta.requires_grad = True

    def bic_correction(self, x):
        """do bic correction
        Args:
            x: a tensor for the result of classification
        """
        count = self.num_init_class

        x_splits = []
        for i in range(self.cur_state):
            x_splits.append(x[:,:, count:count + self.num_new_class[i]])
            count += self.num_new_class[i]

        out = [x[:,:,:self.num_init_class]]
        for i in range(len(x_splits)):
            out.append(self.bias_layers[i](x_splits[i]))

        return torch.cat(out, dim=2)

    def bic_training(self):
        model = self.il_trainer.model
        # freeze all layers
        model.freeze_layers([])
        self.unfreeze()
        is_replay = True

        mean_loss = 0.0
        with torch.cuda.device(0):
            self.optim.zero_grad()
            for iter_num, data in enumerate(self.dataloader_bic):
                loss_info = {}
                
                losses = self.il_loss.forward(data['img'].float().cuda(), data['annot'].cuda(), is_replay=is_replay, is_bic=True)
                loss = torch.tensor(0).float().cuda()
                for key, value in losses.items():
                    if value != None:
                        loss += value
                        loss_info[key] = float(value)
                    else:
                        loss_info[key] = float(0)

                if bool(loss == 0):
                    continue
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()

                # print Info
                output = 'Bic loss | Iter: {0[0]:3d}'
                info = [iter_num]
                for key, value in loss_info.items():
                    output += ' | {0[%d]}: {0[%d]:1.4f}' % (len(info), len(info)+1)
                    info.extend([key, value])
                
                mean_loss += float(loss)
                output += ' | Running loss in Bic: {0[%d]:1.4f}' % (len(info))
                info.append(mean_loss / (iter_num + 1))
                print(output.format(info))

                del losses
        self.freeze()

    def next_state(self):
        self.cur_state += 1
        self.update_tools()


        


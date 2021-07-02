import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import numpy as np
def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

class A_GEM(object):
    def __init__(self, model, replay_dataset, batch_sample = 5):
        self.memory = replay_dataset
        self.model = model
        self.batch_sample = batch_sample
        self.replay_grad = None
        self.update_dataLoader()
        
    def cal_replay_grad(self, optimizer):
        #print("calculate replay grad!")
        self.replay_grad = []
        #data = self.dataLoader.__iter__().next()
        
        prev_status = self.model.distill_loss
        self.model.distill_loss = False
        num_groups = len(self.sampler.groups)
        for group in self.sampler.groups:
            data = []
            for idx in group:
                data.append(self.sampler.data_source[idx])
            data = collater(data)
            temp = []
            try:
                optimizer.zero_grad()

                with torch.cuda.device(0):
                    classification_loss, regression_loss = self.model([data['img'].float().cuda(), data['annot'].cuda()])
                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()
                    loss = classification_loss + regression_loss
                    loss.backward()
                    print('Replay Data: Classification loss: {:1.5f} | Regression loss: {:1.5f}'.format(float(classification_loss), float(regression_loss)))
                    for name, p in self.model.named_parameters():
                        if "prev_model" not in name and "bn" not in name and p.requires_grad:
                            temp.append(p.grad.view(-1))
                    
                    temp = torch.cat(temp) / num_groups
                    if self.replay_grad == []:
                        self.replay_grad = temp
                    else:
                        self.replay_grad += temp
                        
                    del classification_loss, regression_loss
                optimizer.zero_grad()
            except Exception as e:
                print(e)
                continue
        
#             self.cal_replay_grad(optimizer)
        self.model.distill_loss = prev_status
    
    def fix_grad(self):
        # cal current gradient
        cur_grad = []
        for name, p in self.model.named_parameters():
            if "prev_model" not in name and "bn" not in name and p.requires_grad:
                cur_grad.append(p.grad.view(-1))
        cur_grad = torch.cat(cur_grad)
        length_replay = (self.replay_grad * self.replay_grad).sum() # the vector of replay_grad's length 
        angle = (cur_grad * self.replay_grad).sum() # the two gradient's angle
        
        
        # update grad
        if angle < 0:
            proj_grad = cur_grad - ((angle / length_replay) * self.replay_grad) # project gradient
            #proj_grad = torch.where(torch.ge(angle, 0), cur_grad, proj_grad)
            index = 0
            for name, p in self.model.named_parameters():
                if "prev_model" not in name and "bn" not in name and p.requires_grad:
                    n_param = p.numel()  # number of parameters in [p]
                    p.grad.copy_(proj_grad[index:index+n_param].view_as(p))
                    index += n_param
            del proj_grad
        del cur_grad, self.replay_grad
            
    def update_dataLoader(self):
        self.sampler = AspectRatioBasedSampler(self.memory, batch_size = self.batch_sample, drop_last=False)
        #self.dataLoader = DataLoader(self.memory, num_workers=2, collate_fn=collater, batch_sampler=sampler) 


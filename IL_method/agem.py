import torch

from retinanet.dataloader import collater, AspectRatioBasedSampler
from train.train import fast_zero_grad

class A_GEM(object):
    def __init__(self, model, replay_dataset, params):
        self.memory = replay_dataset
        self.model = model
        self.params = params

        # use all data
        self.batch_sample = params['agem_batch']

        self.replay_grad = None
        self.update_dataLoader()
        
    def cal_replay_grad(self):

        self.replay_grad = []

        num_groups = len(self.sampler.groups)
        for group in self.sampler.groups:
            data = []
            for idx in group:
                data.append(self.sampler.data_source[idx])
            data = collater(data)
            temp = []
            try:
                fast_zero_grad(self.model)

                with torch.cuda.amp.autocast():
                    cls_loss, reg_loss = self.model.cal_simple_focal_loss(data['img'].float().cuda(), data['annot'].cuda(), self.params)
                                
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()
                    loss = cls_loss + reg_loss
                    loss.backward()

                    print('Replay Data: Classification loss: {:1.5f} | Regression loss: {:1.5f}'.format(float(cls_loss), float(reg_loss)))
                    
                    for name, p in self.model.named_parameters():
                        if "bn" not in name and p.requires_grad:
                            temp.append(p.grad.view(-1))
                    
                    temp = torch.cat(temp) / num_groups
                    if self.replay_grad == []:
                        self.replay_grad = temp
                    else:
                        self.replay_grad += temp
                        
                    del cls_loss, reg_loss
            except Exception as e:
                print(e)
                continue

    def fix_grad(self):
        # Get current gradient
        cur_grad = []
        for name, p in self.model.named_parameters():
            if "bn" not in name and p.requires_grad:
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
                if "bn" not in name and p.requires_grad:
                    n_param = p.numel()  # number of parameters in [p]
                    p.grad.copy_(proj_grad[index:index + n_param].view_as(p))
                    index += n_param
            del proj_grad
        del cur_grad, self.replay_grad
            
    def update_dataLoader(self):
        self.sampler = AspectRatioBasedSampler(self.memory, batch_size = self.self.params['batch_size'] ,drop_last=False)
        # num_imgs = len(self.memory)
        # if self.batch_sample != -1:
        #     if not (self.batch_sample % 5 == 0 or self.batch_sample % 4 == 0 or self.batch_sample % 3 == 0):
        #         raise ValueError("Please give the number which can be divided by 5, 4 or 3 for agem batch_sample")
        #     for i in [5, 4, 3]:
        #         if self.batch_sample % i == 0:
        #             batch_size = i
        #             break
        #     self.iter_num = int(self.batch_sample / batch_size)
        # else:
        #     batch_size =self.params['batch_size']
        #     self.iter_num = int((num_imgs - 1) / batch_size) + 1
        


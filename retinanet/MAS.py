import torch.nn as nn
import pickle
import os
import torch
class MAS(object):
    def __init__(self, model: nn.Module, dataloader):
        self.model = model
        self.dataloader = dataloader
    def load_importance(self, path):
        pickle_name = "MAS_logits.pickle"
        with open(os.path.join(path, pickle_name), "rb") as f:
            self.precision_matrices = pickle.load(f)
    def calculate_importance(self):
        print('Computing MAS')
        
        origin_status = self.model.distill_feature
        self.model.distill_feature = True
        precision_matrices = {}
        for n, p in self.model.named_parameters():
            if "prev_model" not in n:
                precision_matrices[n] = p.clone().detach().fill_(0)
        self.model.train()
        self.model.freeze_bn()
        num_data = len(self.dataloader)
        for idx, data in enumerate(self.dataloader):
            with torch.cuda.device(0):
                self.model.zero_grad()

                features, regression, classification = self.model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
                                                          
                classification = torch.norm(classification)
                regression = torch.norm(regression)
                regression *= float(classification / regression) 
                output = classification + regression
                output = classification
                output.backward()
                                          
                for n, p in self.model.named_parameters():
                    if p.grad != None:
                        precision_matrices[n].data += p.grad.abs() / num_data

        self.model.distill_feature = origin_status
        self.precision_matrices = precision_matrices

    def penalty(self, model: nn.Module):
        assert model.prev_model != None
        loss = 0
        old_params = {n:p for n,p in model.prev_model.named_parameters()}
        used_names = [name for name in self.precision_matrices.keys()]
        for n, p in model.named_parameters():
            if "classificationModel.output" not in n and "prev_model" not in n and n in used_names:
                _loss = self.precision_matrices[n] * (p - old_params[n]) ** 2
                loss += _loss.sum()
#             else:
#                 _loss = self.precision_matrices[n] * (p[] - self.old_params[n]) ** 2                                                           
#                 self.output.weight.data[i*20 + j,:,:,:] = old_output.weight.data[i*4 + part_idx,:,:,:]
#                 self.output.bias.data[i*20 + j] = old_output.bias.data[i*4 + part_idx]     
        return loss
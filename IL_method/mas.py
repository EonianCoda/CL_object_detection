import pickle
import os
import torch
from preprocessing.debug import debug_print

class MAS(object):
    def __init__(self, model, params):
        self.model = model
        self.params = params
    def load_importance(self, file):
        if not os.path.isfile(file):
            return False
        with open(file, "rb") as f:
            self.precision_matrices = pickle.load(f)
        return True

    def calculate_importance(self, dataloader):
        debug_print('Computing MAS!')
        
        precision_matrices = {}
        for name, params in self.model.named_parameters():
            precision_matrices[name] = params.clone().detach().fill_(0)

        self.model.train()
        self.model.freeze_bn()
        num_data = len(self.dataloader)
        for idx, data in enumerate(dataloader):
            with torch.cuda.device(0):
                self.model.zero_grad()

                classification, regression = self.model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0),
                                                            return_feat=False, 
                                                            return_anchor=False, 
                                                            enable_act=True)
               
                classification = torch.norm(classification)
                regression = torch.norm(regression)
                regression *= float(classification / regression) 
                output = classification + regression
                output = classification
                output.backward()
                                          
                for name, params in self.model.named_parameters():
                    if params.grad != None:
                        precision_matrices[name].data += params.grad.abs() / num_data
    
        self.precision_matrices = precision_matrices

    def penalty(self, prev_model):
        loss = torch.tensor(0).float().cuda()
        old_params = {n:p for n,p in prev_model.named_parameters()}
        used_names = [name for name in self.precision_matrices.keys()]
        for name, params in self.model.named_parameters():
            if "classificationModel.output" not in name and name in used_names:
                _loss = self.precision_matrices[name] * (params - old_params[name]) ** 2
                loss += _loss.sum()
#             else:
#                 _loss = self.precision_matrices[n] * (p[] - self.old_params[n]) ** 2                                                           
#                 self.output.weight.data[i*20 + j,:,:,:] = old_output.weight.data[i*4 + part_idx,:,:,:]
#                 self.output.bias.data[i*20 + j] = old_output.bias.data[i*4 + part_idx]     
        return loss
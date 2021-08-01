import pickle
import os
from matplotlib.pyplot import fill
import torch
import torch.nn as nn
from preprocessing.debug import debug_print, DEBUG_FLAG

FILE_NAME = "mas_importance.pickle"


def fast_zero_grad(model):
    for param in model.parameters():
        param.grad = None

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class Output_norm(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations):
        batch_size = classifications.shape[0]
        result = dict()

        result['regression'] = torch.tensor(0).float().cuda()
        # result['classification'] = torch.tensor(0).float().cuda()
        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # shape=(num_anchors, num_annotations)
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # shape=(num_anchors x 1)
            
            positive_indices = torch.ge(IoU_max, 0.5) 

            # compute the loss for regression
            if positive_indices.sum() > 0:
                regression = regression[positive_indices, :]
                result['regression'] += torch.mean(regression.abs())
                
                
                # result['classification'] += torch.sum(torch.pow(classification, 2))
        
        result['regression'] /= batch_size
        # result['classification'] /= batch_size
        #torch.sum(classifications.pow_(2),dim=1)
        

        result['classification'] = torch.sum(torch.pow(classifications, 2)) / (batch_size * classifications.shape[2])
        return result


class MAS(object):
    def __init__(self, model, params):
        self.model = model
        self.params = params
    def load_importance(self, state:int):
        """load the impotance pickle file
            Args:
                state:int
        """
        path = os.path.join(self.params['ckp_path'], "state{}".format(state))
        file_name = FILE_NAME

        # File isn't exist
        if not os.path.isfile(os.path.join(path, file_name)):
            return False


        with open(os.path.join(path, file_name), "rb") as f:
            self.precision_matrices = pickle.load(f)
        return True

    def calculate_importance(self, dataloader, state:int):
        debug_print('Computing MAS!')
        
        precision_matrices = {}
        for name, p in self.model.named_parameters():
            if "bn" not in name and p.requires_grad and "classificationModel.output" not in name:
                precision_matrices[name] = p.clone().detach().fill_(0)

        self.model.train()
        self.model.freeze_bn()
        num_batch = len(dataloader)
        output_norm = Output_norm()

        for idx, data in enumerate(dataloader):
            with torch.cuda.device(0):
                fast_zero_grad(self.model)
                try:
                    classifications, regressions, anchors=  self.model(data['img'].float().cuda(),
                                                                return_feat=False, 
                                                                return_anchor=True, 
                                                                enable_act=True)
                    norms = output_norm(classifications, regressions, anchors, data['annot'].cuda())
                    output = norms['classification'] + norms['regression']
                    output.backward()
                    print(idx)
                    for name, p in self.model.named_parameters():
                        if "bn" not in name and p.requires_grad and "classificationModel.output" not in name:
                            precision_matrices[name].data += p.grad.abs()
                except Exception as e:
                    self.num_batch -= 1
                    print(e)
                    continue
        for key in precision_matrices:
            precision_matrices[key] /= num_batch


        path = os.path.join(self.params['ckp_path'], "state{}".format(state))
        file_name = FILE_NAME
        with open(os.path.join(path, file_name), "wb") as f:
            pickle.dump(precision_matrices, f)

    def penalty(self, prev_model, mas_ratio=1.0):
        loss = torch.tensor(0).float().cuda()
        old_params = {n:p for n,p in prev_model.named_parameters()}
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.precision_matrices.keys():
                temp = self.precision_matrices[name] * (p - old_params[name]) ** 2
                loss += temp.sum() * mas_ratio

        return loss

    def destroy(self):
        del self.precision_matrices
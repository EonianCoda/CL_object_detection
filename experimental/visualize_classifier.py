from _typeshed import Self

from torch._C import set_flush_denormal
from main import get_parser, Params
import torch
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from collections import defaultdict

class Visualizer(object):
    def __init__(self, params):
        self.params = params
        self.model = None

    def set_model(self, state:int, epoch:int):
        if self.model:
            del self.model
        self.model = self.params.get_model(state, epoch)
        self.state = state
        self.epoch = epoch

        self.classifier = self._get_classifier_weights(self.model)


    def _get_classifier_weights(model):
        """get classifier weight from the model
            Args:
                model: the retinanet model
        """
        num_classes = model.num_classes
        num_anchors = model.classificationModel.num_anchors

        # output_weight = self.get_parameters("classificationModel.output.weight")
        # output_bias = self.get_parameters("classificationModel.output.bias")
        output_weight = model.state_dict()["classificationModel.output.weight"]
        output_bias = model.state_dict()["classificationModel.output.bias"]
        
        classed_parameters = [{'weight':[],'bias':[]} for _ in range(num_classes)]

        
        for i in range(num_anchors):
            start_idx = i * num_classes
            for class_idx in range(num_classes):
                classed_parameters[class_idx]['weight'].append(output_weight.data[start_idx + class_idx,:,:,:])
                classed_parameters[class_idx]['bias'].append(output_bias.data[start_idx + class_idx])
        for class_idx in range(num_classes):
            classed_parameters[class_idx]['weight'] = torch.cat(classed_parameters[class_idx]['weight'])
            classed_parameters[class_idx]['bias'] = torch.tensor(classed_parameters[class_idx]['bias'])

        return classed_parameters
    
    def get_weight_norms(self):
        """
            Args:
                state: int, the prefered state
                epoch: int, the prefered epoch
        """
        def cal_norm(x):
            return float(torch.norm(x))

        if not self.model:
            raise ValueError("Please call set model first!")

        weight_norms = []
        bias_norms = []
        for class_id, data in enumerate(self.classifier):
            weight_norms.append(cal_norm(data['weight']))
            bias_norms.append(cal_norm(data['bias']))
        return weight_norms, bias_norms
    
    def _cal_ranked_mean(classifier, start_cid, end_cid, smooth = 1):
        ranked_values = []
        num_classes = end_cid - start_cid

        for cid in range(start_cid, end_cid):
            weight = classifier[cid].flatten()
            values, indices = torch.sort(weight, descending=True)
            classed_values = []
            for start_idx in range(0, len(values), smooth):
                temp = 0
                for i in range(smooth):
                    temp += values[start_idx + i] / smooth

                classed_values.append((values[i] + values[i+1]) / smooth)

            classed_values = np.array(classed_values)
            if len(ranked_values) == 0:
                ranked_values = classed_values / num_classes
            else:
                ranked_values += classed_values / num_classes
        return ranked_values

    def get_ranked_mean_weights(self, smooth=8):
        if not self.model:
            raise ValueError("Please call set model first!")
        
        num_new_class = self.arams.states[self.state]['num_new_class']
        num_old_class = self.params.states[self.state]['num_past_class']

        old_ranked_mean = self._cal_ranked_mean(self.classifier,0, num_old_class,smooth)
        new_ranked_mean = self._cal_ranked_mean(self.classifier,num_old_class , num_old_class,smooth)
        return old_ranked_mean


# def show(start_epoch:int, end_epoch:int, state:int):
#     plt.figure(figsize=(8,8))
#     ax = plt.gca()
#     ax.set_prop_cycle(cycler('color', color) +
#                        cycler('lw', [1] * len(color)))
    
#     plt.title('2-norm of classifier weight for state{}'.format(state))
#     plt.xlabel('class id')
#     plt.ylabel('2-norm of weight')
#     for epoch in range(start_epoch, end_epoch + 1, 5):
#         weight_norms, bias_norms = get_classifier_norm(state, epoch)
#         # weight_norms = weight_norms[:-1]
#         plt.plot(range(1, len(weight_norms) + 1), weight_norms, label=str(epoch))
    
#     #weight_norms, bias_norms = get_classifier_norm(0, -1)
    
    
    
#     plt.legend()
        



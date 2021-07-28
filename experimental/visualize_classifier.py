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
        self.num_new_class = self.params.states[self.state]['num_new_class']
        if self.state != 0:
            self.num_old_class = self.params.states[self.state]['num_past_class']


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
    
    def _get_weight_norms(self):
        def cal_norm(x):
            return float(torch.norm(x))

        if not self.model:
            raise ValueError("Please call set model first!")

        weight_norms = dict()
        bias_norms = dict()
        for class_id, data in enumerate(self.classifier):
            weight_norms['class_id'] = cal_norm(data['weight'])
            bias_norms['class_id'] = cal_norm(data['bias'])
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

    def _get_ranked_mean_weights(self, smooth=8):
        if not self.model:
            raise ValueError("Please call set model first!")
        
        if self.state != 0:
            old_ranked_mean = self._cal_ranked_mean(self.classifier,0, self.num_old_class,smooth)
        new_ranked_mean = self._cal_ranked_mean(self.classifier, self.num_old_class, self.num_old_class + self.num_new_class,smooth)
        
        if self.state != 0:
            return old_ranked_mean, new_ranked_mean
        else:
            return new_ranked_mean

    def _create_fig(figsize=(8,8)):
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        color = ['b','g','r','c', 'm', 'y', 'k']
        ax.set_prop_cycle(cycler('color', color) + cycler('lw', [1] * len(color)))
        # return fig
    def show_ranked_mean_weight(self, smooth=8):
        if self.state == 0:
            new_ranked_mean = self._get_ranked_mean_weights(smooth)
        else:
            old_ranked_mean, new_ranked_mean = self._get_ranked_mean_weights(smooth)

        # show figure
        self._create_fig((8,12))

        plt.title('mean of ranked weight of classifier for state{}'.format(self.state))
        plt.bar(range(len(new_ranked_mean)), new_ranked_mean, label='new task')
        if self.state != 0:
            plt.bar(range(len(old_ranked_mean)), old_ranked_mean, label='old task')
        plt.xlabel('ranked')
        plt.ylabel('mean of ranked weight')
        plt.legend()
        plt.show()

    def show_weight_norm(self):
        weight_norms, bias_norms = self._get_weight_norms()


        self._create_fig((8,8))

        plt.title('norm for weight of classifier for state{}'.format(self.state))
        cat_ids = [cat_id for cat_id in weight_norms.keys()]
        cat_ids.sort()
        
        weight_norms_list = []
        for cat_id in cat_ids:
            weight_norms_list.append(weight_norms[cat_id])

        plt.plot(range(1, len(weight_norms_list)+1), weight_norms_list)
        plt.xlabel('class_id')
        plt.ylabel('norm of classifier weigt')
        plt.legend()
        plt.show()
        
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
        



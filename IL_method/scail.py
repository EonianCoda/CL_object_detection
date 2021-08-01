from preprocessing.params import Params
from experimental.visualize_classifier import get_classifier_weights
import torch

#TODO

def ranked_mean_data(classifier, num_classes:int):
    def cal_mean(key):
        classifier_weights = [data[key].flatten() for data in classifier]
        classifier_weights = classifier_weights[:num_classes]
        # sort weight
        for cat_id in range(len(classifier_weights)):
            value, _ = classifier_weights[cat_id].sort()
            classifier_weights[cat_id] = value.unsqueeze(dim=0)

        classifier_weights = torch.cat(classifier_weights)

        mean_weight = classifier_weights.abs().mean(dim=0)
        return mean_weight

    mean_weights = cal_mean('weight')
    mean_biases = cal_mean('bias')

    return mean_weights, mean_biases

class Scail(object):
    def __init__(self, params:Params, cur_state:int):
        self.params = params
        self.cur_state = cur_state




    # def _get_scaled_classifier(self ):
        
    # def get_scaled_mode(self):
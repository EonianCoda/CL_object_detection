import torch
from retinanet.model import ResNet
from preprocessing.params import Params
# VAL_ARG_SAMPLE = "--dataset voc2007 \
#                     --state 1 \
#                     --epoch 30 \
#                     --scenario 15 1 \
#                     --just_val True"
                    
# TRAIN_ARG_SAMPLE = "--dataset voc2007 \
#                     --start_epoch 1 \
#                     --end_epoch 100 \
#                     --start_state 0 \
#                     --end_state 0 \
#                     --scenario 10 10\
#                     --print_il_info True\
#                     --debug False \
#                     --record False"

def text_to_args(args):
    """convert text to args, in order to create Params
    """
    args = [arg.rstrip() for arg in args.split('--') if arg != '']
    result_arg = []
    for arg in args:
        texts = arg.split(' ')
        result_arg.append('--' + texts[0])
        for i in range(1, len(texts)):
            result_arg.append(texts[i])
    return result_arg


class Experiment_tool(object):
    def __init__(self, model:ResNet, params: Params):
        self.model = model
        self.params = params
        self.num_classes = self.model.num_classes()
        self._init_layer_info()

    def _init_layer_info(self):
        self.layer_names = []
        self.parameters = dict()
        for name, p in self.model.named_parameters():
            self.layer_names.append(name)
            self.parameters[name] = p

    def get_parameters(self, name):
        return self.parameters[name]

    def get_classed_classifier(self):
        num_anchors = self.model.classificationModel.num_anchors
        output_weight = self.get_parameters("classificationModel.output.weight")
        output_bias = self.get_parameters("classificationModel.output.bias")

        class_ids = self.params.states[-1]['num']['knowing_class']['id'][:self.num_classes]
        class_names = self.params.states.coco.catId_to_name(class_ids)

        classed_parameters = [{'weight':[],'bias':[]} for _ in range(self.num_classes)]
        for i in range(num_anchors):
            start_idx = i * self.num_classes
            for class_idx in range(self.num_classes):
                classed_parameters[class_idx]['weight'].append(output_weight.data[start_idx + class_idx,:,:,:])
                classed_parameters[class_idx]['bias'].append(output_bias.data[start_idx + class_idx])
        for class_idx in range(self.num_classes):
            classed_parameters[class_idx]['weight'] = torch.cat(classed_parameters[class_idx]['weight'])
            classed_parameters[class_idx]['bias'] = torch.cat(classed_parameters[class_idx]['bias'])

        return classed_parameters
            # self.output.bias.data[i * self.num_classes:i * self.num_classes + old_classes] = old_output.bias.data[i * old_classes:(i+1) * old_classes]
            



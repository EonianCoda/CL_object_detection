import torch


class Experimental_tool(object):
    def __init__(self, model, params):
        self.model = model
        self.params = params
        self.num_classes = self.model.num_classes
        self._init_layer_info()

    def _init_layer_info(self):
        self.layer_names = []
        self.parameters = dict()
        for name, p in self.model.named_parameters():
            self.layer_names.append(name)
            self.parameters[name] = p

    def get_parameters(self, name):
        return self.parameters[name]

    def get_classified_classifier(self):
        num_anchors = self.model.classificationModel.num_anchors
        output_weight = self.get_parameters("classificationModel.output.weight")
        output_bias = self.get_parameters("classificationModel.output.bias")

        class_ids = self.params.states[-1]['knowing_class']['id'][:self.num_classes]
        class_names = self.params.states.coco.catId_to_name(class_ids)

        classed_parameters = [{'weight':[],'bias':[]} for _ in range(self.num_classes)]
        for i in range(num_anchors):
            start_idx = i * self.num_classes
            for class_idx in range(self.num_classes):
                classed_parameters[class_idx]['weight'].append(output_weight.data[start_idx + class_idx,:,:,:])
                classed_parameters[class_idx]['bias'].append(output_bias.data[start_idx + class_idx])
        for class_idx in range(self.num_classes):
            classed_parameters[class_idx]['weight'] = torch.cat(classed_parameters[class_idx]['weight'])
            classed_parameters[class_idx]['bias'] = torch.tensor(classed_parameters[class_idx]['bias'])

        return classed_parameters

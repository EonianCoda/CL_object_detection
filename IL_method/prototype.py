
from torch.nn import functional as F
import torch
import torch.nn as nn
import os
import pickle

from torch.utils.data.dataloader import DataLoader

from torchvision import transforms

# my package
from retinanet.losses import calc_iou
from retinanet.dataloader import IL_dataset, Resizer, Augmenter, Normalizer, collater, AspectRatioBasedSampler
from preprocessing.params import create_dir

class ProtoTyper(object):
    def __init__(self, il_trainer, thresold = 0.5):
        self.il_trainer = il_trainer
        self.thresold = thresold
        self.num_anchors = self.il_trainer.model.classificationModel.num_anchors
        self.prototype_features = None

    def _get_positive(self, anchors, annotations):

        batch_size = annotations.shape[0]
        
        positive_indices = []
        targets = []
        
        for j in range(batch_size):
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # shape=(num_anchors, num_annotations)
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # shape=(num_anchors x 1)
            
            pos_indices = torch.ge(IoU_max, self.thresold).view(-1, self.num_anchors)
            target = bbox_annotation[IoU_argmax, 4].view(-1, self.num_anchors)
            
            #store the postive anchor and its class
            positive_indices.append(pos_indices.unsqueeze(dim=0))
            targets.append(target.long().unsqueeze(dim=0))
        
        positive_indices = torch.cat(positive_indices)
        targets = torch.cat(targets)

        return positive_indices, targets

    def _cal_features(self, feature_temp_path:str, state:int):
        """calculate the features for each image and store in  feature_temp_path
        Args:
            feature_temp_path(str): the path for store the features for each image
        """

    
        dataset = IL_dataset(self.il_trainer.params,
                    transform=transforms.Compose([Normalizer(), Resizer()]),
                    start_state=state)

        # create the dataloader for cal the features
        sampler = AspectRatioBasedSampler(dataset, batch_size = self.il_trainer.params['batch_size'], drop_last=False, shuffle=False)
        dataloader = DataLoader(dataset, num_workers=2, collate_fn=collater, batch_sampler=sampler)
        model = self.il_trainer.model
        num_classes = model.classificationModel.num_classes

        batch_size = self.il_trainer.params['batch_size']
        for idx, data in enumerate(dataloader):
            with torch.no_grad():
                img_batch = data['img'].float().cuda()
                annot = data['annot'].cuda()
                
                # get features from the classification head
                features, anchors = model.get_classification_feature(img_batch)
                positive_indices, targets = self._get_positive(anchors, annot)

                  
                for batch_id in range(batch_size):
                    # init data for each img
                    count = torch.zeros(num_classes, self.num_anchors, 1).cuda()
                    prototype_features = torch.zeros(num_classes, self.num_anchors, 256 * self.num_anchors).cuda()
                    
                    iter_num = idx * batch_size + batch_id
                    
                    # get each img's data in minibatch
                    feature = features[batch_id,...]
                    pos = positive_indices[batch_id,...]
                    target = targets[batch_id,...]
                    
                    # get the positive anchor
                    mask = pos.any(dim=1)
                    feature = feature[mask,:]
                    pos = pos[mask,:]
                    target = target[mask,:]

                    # accumulate the features and num anchors
                    for i in range(feature.shape[0]):
                        count[target[i][pos[i]], pos[i],:] += 1
                        prototype_features[target[i][pos[i]], pos[i],:] += feature[i]
                    
                    # store the features and positive anchors's number
                    with open(os.path.join(feature_temp_path, 'f_{}.pickle'.format(iter_num)), 'wb') as f:
                        pickle.dump((prototype_features.cpu(), count.cpu()), f)

        del dataloader
        del dataset

    def init_prototype(self, state:int): 
        num_classes = self.il_trainer.model.classificationModel.num_classes
        feature_temp_path = os.path.join(self.il_trainer.params['ckp_path'], 'state{}'.format(state), 'features')
        create_dir(feature_temp_path)


        path = os.path.join(self.il_trainer.params['ckp_path'], 'state{}'.format(state))
        file_name = "prototype_features.pickle"
        if os.path.isfile(os.path.join(path, file_name)):
            # load the prototype
            with open(os.path.join(path, file_name), "rb") as f:
                self.prototype_features = pickle.load(f)
        else:
            # calculate the features for all training data
            self._cal_features(feature_temp_path, state)

            # use the temp file for features to calculate the prototype
            count = torch.zeros(num_classes, self.num_anchors, 1)
            self.prototype_features = torch.zeros(num_classes, self.num_anchors, 256 * self.num_anchors)  #TODO 256 auto get

            num_files = len(os.listdir(feature_temp_path))

            for i in range(num_files):
                with open(os.path.join(feature_temp_path,'f_{}.pickle'.format(i)), 'rb') as f:
                    _ , num = pickle.load(f)
                    count += num

            for i in range(num_files):
                with open(os.path.join(feature_temp_path,'f_{}.pickle'.format(i)), 'rb') as f:
                    feat, _ = pickle.load(f)
                    self.prototype_features += (feat / torch.clamp(count, min=1))

            # store the prototype
            with open(os.path.join(path, file_name), "wb") as f:
                pickle.dump(self.prototype_features, f)

    def cal_examplar(self, state:int):
        def distance_fun(a, b):
            return torch.norm(a - b, dim=3)


        file_path = os.path.join(self.il_trainer.params['ckp_path'], 'state{}'.format(state))
        file_name = "classification_herd_samples.pickle"
        # if temp file exists, then  return
        if os.path.isfile(os.path.join(file_path, file_name)):
            return


        feature_temp_path = os.path.join(self.il_trainer.params['ckp_path'], 'state{}'.format(state), 'features')
        create_dir(feature_temp_path)


        num_classes = len(self.il_trainer.params.states[state]['knowing_class']['id'])
        num_new_classes = len(self.il_trainer.params.states[state]['new_class']['id'])
        if num_classes != self.il_trainer.model.classificationModel.num_classes:
            raise ValueError("Current model has {} classes, but state file has {} classes.".format(self.il_trainer.model.classificationModel.num_classes, num_classes))

        

        # get the number of the feature files
        num_files = len(os.listdir(feature_temp_path))
        # if the feature temp file not exits, then calculate features
        if num_files == 0 or self.prototype_features == None:
            self.init_prototype(state)
            if num_files == 0:
                num_files = len(os.listdir(feature_temp_path))
                if num_files == 0:
                    raise ValueError("Unknowing Error in cal_examplar")

        feats = []
        count = torch.zeros(num_classes, self.num_anchors, 1)
        for i in range(num_files):
            with open(os.path.join(feature_temp_path,'f_{}.pickle'.format(i)), 'rb') as f:
                feat, num = pickle.load(f)
                feats.append((feat / torch.clamp(num, min=1)).unsqueeze(dim=0))
                count += num

        feats = torch.cat(feats)

        has_target_mask = ~(torch.sum(feats, dim=3) == 0)
        distance_target = torch.zeros_like(has_target_mask).long()
        distance_target[has_target_mask] = 1
        distance = distance_fun(feats, self.prototype_features.unsqueeze(dim=0))
        distance *= distance_target


        dataset = IL_dataset(self.il_trainer.params,
                            transform=transforms.Compose([Normalizer(), Resizer()]),
                            start_state=state)
        sampler = AspectRatioBasedSampler(dataset, batch_size = self.il_trainer.params['batch_size'], drop_last=False, shuffle=False)
        # mapping index to real image id
        img_ids = []
        for group in sampler.groups:
            img_ids.extend(group)
        for i in range(len(img_ids)):
            img_ids[i] = dataset.image_ids[img_ids[i]]
        img_ids = torch.tensor(img_ids)



        sample_file = dict()
        
        for class_id in range(num_classes - num_new_classes, num_classes):
            coco_id = dataset.label_to_coco_label(class_id)
            sample_file[coco_id] = dict()
            for anchor_id in range(self.num_anchors):
                cur_distance = distance[:,class_id, anchor_id]
                nonzero_ids = cur_distance.nonzero().squeeze()
                sorted_ids = nonzero_ids[cur_distance[nonzero_ids].argsort()] # nonzero_ids.gather(0, cur_distance[nonzero_ids].argsort())
                sorted_ids = img_ids[sorted_ids]
                sample_file[coco_id][anchor_id] = sorted_ids.tolist()

        with open(os.path.join(file_path, file_name),'wb') as f:
            pickle.dump((sample_file, count), f)

    def __del__(self):
        if self.prototype_features != None:
            del self.prototype_features
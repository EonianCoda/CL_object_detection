
from torch.nn import functional as F
import torch
import torch.nn as nn
import os
import pickle

from torch.utils.data.dataloader import DataLoader

from torchvision import transforms

# my package
from retinanet.losses import calc_iou
from retinanet.dataloader import Resizer, Augmenter, Normalizer, collater, AspectRatioBasedSampler
from preprocessing.params import create_dir

class ProtoTyper(object):
    def __init__(self, il_trainer, thresold = 0.5):
        self.il_trainer = il_trainer
        self.thresold = thresold
        self.num_anchors = self.il_trainer.model.classificationModel.num_anchors

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

    def _cal_features(self, feature_temp_path:str):
        """calculate the features for each image and store in  feature_temp_path
        Args:
            feature_temp_path(str): the path for store the features for each image
        """
        dataset = self.il_trainer.dataset_train
        

        # close the augumenter
        dataset.transform = transforms.Compose([Normalizer(), Resizer()])

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

        # recover the transform
        dataset.transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
        del dataloader

    def init_prototype(self, state:int):
        
        num_classes = self.il_trainer.model.classificationModel.num_classes
        feature_temp_path = os.path.join(self.il_trainer.params['ckp_path'], 'state{}'.format(state), 'features')
        create_dir(feature_temp_path)


        path = os.path.join(self.il_trainer.params['ckp_path'], 'state{}'.format(state))
        file_name = "prototype_features.pickle"
        if os.path.isfile(os.path.join(path, file_name)):
            with open(os.path.join(path, file_name), "rb") as f:
                self.prototype_features = pickle.load(f)
            #self.prototype_features = self.prototype_features.unsqueeze(dim = 0)
        else:
            # calculate the features for all training data
            self._cal_features(feature_temp_path)

            # use the temp file for features to calculate the prototype
            count = torch.zeros(num_classes, self.num_anchors, 1)
            self.prototype_features = torch.zeros(num_classes, self.num_anchors, 256 * self.num_anchors)  #TODO 256 auto get

            num_files = len(os.listdir(feature_temp_path))

            for i in range(num_files):
                with open(os.path.join(feature_temp_path,'f{}.pickle'.format(i)), 'rb') as f:
                    _ , num = pickle.load(f)
                    count += num

            for i in range(num_files):
                with open(os.path.join(feature_temp_path,'f{}.pickle'.format(i)), 'rb') as f:
                    feat, _ = pickle.load(f)
                    self.prototype_features += (feat / torch.clamp(count, min=1))

            # store the prototype
            with open(os.path.join(path, file_name), "wb") as f:
                pickle.dump(self.prototype_features, f)

    def cal_examplar(self, state:int):
        def distance_fun(a, b):
            return torch.norm(a - b, dim=3)


        path = os.path.join(self.il_trainer.params['ckp_path'], 'state{}'.format(state))
        file_name = "classification_herd_samples.pickle"

        if os.path.isfile(os.path.join(path, file_name)):
            return


        feature_temp_path = os.path.join(self.il_trainer.params['ckp_path'], 'state{}'.format(state), 'features')
        create_dir(feature_temp_path)
        num_classes = self.il_trainer.model.classificationModel.num_classes


        feats = []
        count = torch.zeros(num_classes, self.num_anchors, 1)

        num_files = len(os.listdir(feature_temp_path))
        if num_files == 0:
            raise ValueError("The feature temp file isn't exist, please call _cal_features first!")
        for i in range(num_files):
            with open(os.path.join(feature_temp_path,'f{}.pickle'.format(i)), 'rb') as f:
                feat, num = pickle.load(f)
                feats.append((feat / torch.clamp(num, min=1)).unsqueeze(dim=0))
                count += num

        feats = torch.cat(feats)

        pos_mask = ~(torch.sum(feats, dim=3) == 0)
        distance_target = torch.zeros_like(pos_mask).long()
        distance_target[pos_mask] = 1
        distance = distance_fun(feats, self.prototype_features.unsqueeze(dim=0))
        distance *= distance_target



        sampler = AspectRatioBasedSampler(self.il_trainer.dataset_train, batch_size = self.il_trainer.params['batch_size'], drop_last=False, shuffle=False)
        img_ids = []
        for g in sampler.groups:
            img_ids.extend(g)
        for i in range(len(img_ids)):
            img_ids[i] = self.il_trainer.dataset_train.image_ids[img_ids[i]]


        sample_file = {}
        for class_id in range(num_classes):

            sample_file[class_id] = {}
            for anchor_id in range(self.num_anchors):
                cur_distance = distance[:,class_id, anchor_id]
                nonzero_ids = cur_distance.nonzero().squeeze()
                sorted_ids = nonzero_ids.gather(0, cur_distance[nonzero_ids].argsort())
                for i in range(len(sorted_ids)):
                    sorted_ids[i] = img_ids[sorted_ids[i]]

                sample_file[class_id][anchor_id] = sorted_ids.tolist()




        with open(os.path.join(path, file_name),'wb') as f:
            pickle.dump((sample_file, count), f)
from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
import pickle
#import cocoAPI
from collections import defaultdict
from PIL import Image
import random

class cocoAPI():
    def __init__(self, coco_path):
        self.coco = COCO(coco_path)

        self.classes = defaultdict()
        self.reverse_classes = defaultdict()
        for category in self.coco.loadCats(self.coco.getCatIds()):
            self.classes[category['id']] = category['name']
            self.reverse_classes[category['name']] = category['id']

    def getImgCats(self, imgIds, return_name=False):

        #get annotations
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=imgIds))
        #get categoryId appear in annotations
        catIds = [ann['category_id'] for ann in annotations]
        catIds = list(set(catIds))

        if not return_name:
            return catIds
        else:
            catNames = []
            #get category name from categord Id
            for catId in catIds:
                catNames.append(self.classes[catId])
            return catNames

    def getImgIdFromCats(self, catIds):
        if type(catIds) == list:
            imgIds = set()
            for catId in catIds:
                imgIds.update(self.coco.getImgIds(catIds=catId))
            return list(imgIds)
        else:
            return self.coco.getImgIds(catIds=catIds)

    def catIdToName(self, catIds):
        if type(catIds) == int:
            return [self.classes[catIds]]

        else:
            names = [self.classes[catId] for catId in catIds]
            return names
                
    def catNameToId(self, names):
        if type(names) == str:
            return [self.reverse_classes[names]]

        names = list(set(names))

        ids = []
        for name in names:
            ids.append(self.reverse_classes[name])
        ids.sort()

        return ids
    
    def getCatNumfromCatId(self, catIds):
        result = {'image':[], 'object':[]}
        index = []
        catIds.sort()
        
        for catId in catIds:
            index.append(self.classes[catId])
            result['image'].append(len(self.coco.getImgIds(catIds = catId)))
            result['object'].append(len(self.coco.getAnnIds(catIds = catId)))

        index.append('Counts')
        result['image'].append(sum(result['image']))
        result['object'].append(sum(result['object']))
        result = pd.DataFrame(result, index=index)
        result.sort_values(by=['image'], ascending=False)
        return result

    def getCatNumfromImgId(self, imgIds):
        #get annotations
        annotations = self.coco.loadAnns(self.coco.getAnnIds(imgIds=imgIds))
        #get categoryId appear in annotations
        catIds = [ann['category_id'] for ann in annotations]

        result = {'image':[], 'object':[]}

        index = self.catIdToName(list(set(catIds)))
        #object counts
        result['object'] = np.unique(catIds, return_counts=True)[1].tolist()

        catIds = list(set(catIds))
        for catId in catIds:
            result['image'].append(len(set(self.coco.getImgIds(catIds = catId)) & set(imgIds)))
        
        print('Counts meaning for  image is your input imgIds number')
        index.append('Counts')
        result['image'].append(len(imgIds))
        result['object'].append(sum(result['object']))
        result = pd.DataFrame(result, index=index)
        result.sort_values(by=['image'], ascending=False)
        return result


class CocoDataset_inOrder(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='TrainVoc2012', dataset = 'voc', transform=None, start_round = 1, data_split="10+10", rehearsal= 'none',test_flag=True, custom_ids=[]):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.dataset = dataset
        self.prev_imgIds = []

        #read annotation and some related data
        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        
        self.cocoHelper = cocoAPI(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json')) #my custom cocoAPI

        self.classOrder = {'id':[],'name':[]}
        cat_names_sorted = sorted(self.cocoHelper.classes.values())
        if data_split == "20":
            self.classOrder['name'] = [cat_names_sorted]
        elif data_split == "10+10":
            self.classOrder['name'] = [cat_names_sorted[:10], cat_names_sorted[10:]]
        elif data_split == "19+1":
            self.classOrder['name'] = [cat_names_sorted[:19], [cat_names_sorted[19:]]]
        elif data_split == "15+1":
            self.classOrder['name'] = [cat_names_sorted[:15], [cat_names_sorted[18]], [cat_names_sorted[16]], 
                                       [cat_names_sorted[17]], [cat_names_sorted[15]], [cat_names_sorted[19]]]
        elif data_split == "custom":
            """
            ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
            """
            custom = [['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow','diningtable', 'dog', 'horse', 'motorbike', 'person','train'],[ 'pottedplant', 'sheep', 'sofa','tvmonitor']]
#             custom = [['pottedplant','diningtable','boat','cow','cat'], 
#                         ['dog', 'horse', 'motorbike', 'person', 'sofa', 'train', 'tvmonitor']]
            self.classOrder['name'] = custom

        self.classOrder['id'] = [self.cocoHelper.catNameToId(names) for names in self.classOrder['name']]

        print(self.classOrder)
        self.now_round = start_round

        if test_flag:
            if ("Val" in set_name) or ("Test" in set_name):
                self.seen_class_id = []
                for i in range(self.now_round):
                    self.seen_class_id.extend(self.classOrder['id'][i])
            else:
                self.seen_class_id = self.classOrder['id'][self.now_round - 1]
        else:
            if len(custom_ids) == 0:
                self.seen_class_id = self.classOrder['id'][self.now_round - 1]
            else:
                self.seen_class_id = custom_ids

        self.load_classes()
        #get this round's data
        self.update_imgIds()
        
        if dataset == 'coco':
            self.image_path = os.path.join(self.root_dir, 'images', self.set_name)
        elif dataset == 'voc':
            self.image_path = os.path.join(self.root_dir, 'images')
        

    def update_imgIds(self):
        self.image_ids = self.cocoHelper.getImgIdFromCats(self.seen_class_id)
#         if ("Val" in self.set_name) or ("Test" in self.set_name):
#             self.image_ids = self.cocoHelper.getImgIdFromCats(self.seen_class_id)
#         else:
#             imgIds = self.cocoHelper.getImgIdFromCats(self.seen_class_id)
#             self.image_ids = imgIds[:len(imgIds)// 4]
        
        
            

    def next_round(self):
        print('next_round')
        if self.now_round == len(self.classOrder['id']):
            raise(ValueError("Next round doesn't exist."))

        
        self.now_round += 1
        self.seen_class_id = self.classOrder['id'][self.now_round - 1]
        self.update_imgIds()
        
    
    def load_classes(self):

        self.coco_labels         = {} # dataloader ID -> origin annotation ID
        self.coco_labels_inverse = {} # origin annotation ID -> dataloader ID

        idx = 0

#         for order in self.classOrder['id']:
#             for catId in order:
#                 for i in range(4):
#                     if catId in self.supercategory[i]:
#                         self.coco_labels_inverse[catId] = i
#                         break
#         print(self.coco_labels_inverse)
        
        for order in self.classOrder['id']:
            for catId in order:
                self.coco_labels[idx] = catId
                self.coco_labels_inverse[catId] = idx
                idx += 1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        
        
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        file_name = image_info['file_name']
        path = os.path.join(self.image_path, file_name) #file_name[:-4] mean the image's id
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
            
        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, ann in enumerate(coco_annotations):
            #When the category doesn't exist in this round, then ignored it 
            if(ann['category_id'] not in self.seen_class_id):
                continue


            # some annotations have basically no width / height, skip them
            if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = ann['bbox']
            annotation[0, 4]  = self.coco_label_to_label(ann['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        if self.dataset == 'coco':
            return 80
        elif self.dataset == 'voc':
            num = 0
            for r in range(0, self.now_round):
                num += len(self.classOrder['id'][r])
            print('dataloader class_num = {}'.format(num))
            return num
    
class rehearsal_DataSet(Dataset):
    def __init__(self, root_dir, set_name='TrainVoc2012', dataset = 'voc', transform=None, data_split="10+10", method = 'random', now_round = None, per_num = 1, img_ids = []):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        print('init rehearsal_DataSet')
        self.root_dir = root_dir
        

        
        self.set_name = set_name
        self.transform = transform
        self.dataset = dataset
        self.per_num = per_num #how many picutes each class has 
        self.method = method
        self.data_split = data_split
        #read annotation and some related data
        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        
        self.cocoHelper = cocoAPI(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json')) #my custom cocoAPI

        self.classOrder = {'id':[],'name':[]}
        cat_names_sorted = sorted(self.cocoHelper.classes.values())
        if data_split == "20":
            self.classOrder['name'] = [cat_names_sorted]
        elif data_split == "10+10":
            self.classOrder['name'] = [cat_names_sorted[:10], cat_names_sorted[10:]]
        elif data_split == "19+1":
            self.classOrder['name'] = [cat_names_sorted[:19], [cat_names_sorted[19:]]]
        elif data_split == "15+1":
            self.classOrder['name'] = [cat_names_sorted[:15], [cat_names_sorted[18]], [cat_names_sorted[16]], 
                                       [cat_names_sorted[17]], [cat_names_sorted[15]], [cat_names_sorted[19]]]
        elif data_split == "custom":
            """
            ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 
            'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
            """
            custom =  [['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'person', 'chair', 'cow'], 
                        ['diningtable', 'dog', 'horse', 'motorbike', 'train', 'pottedplant', 'sheep', 'sofa', 'cat', 'tvmonitor']]
#             custom = [['pottedplant','diningtable','boat','cow','cat'], 
#                         ['dog', 'horse', 'motorbike', 'person', 'sofa', 'train', 'tvmonitor']]
            self.classOrder['name'] = custom

        self.classOrder['id'] = [self.cocoHelper.catNameToId(names) for names in self.classOrder['name']]

        self.load_classes()
        #get this round's data
        
        
        if dataset == 'coco':
            self.image_path = os.path.join(self.root_dir, 'images', self.set_name)
        elif dataset == 'voc':
            self.image_path = os.path.join(self.root_dir, 'images')
            
            
        self.image_ids = img_ids
        
        if len(self.image_ids) != 0:
            class_num = int(len(self.image_ids) / self.per_num)
            
            for i in range(len(self.classOrder['id'])):
                print(i)
                if class_num == 0:
                    self.now_round = i + 1
                    break
                class_num -= len(self.classOrder['id'][i])
            
            print('rehearsal now_round=',self.now_round)
        else:
            if now_round != None:
                self.reset_by_round(now_round)
            else:
                self.now_round = None
    def reset_by_imgIds(self, per_num = 1, img_ids = []):
                           
        self.image_ids = img_ids
        self.per_num = per_num #how many picutes each class has 
        if len(self.image_ids) != 0:
            class_num = int(len(self.image_ids) / self.per_num)

            for i in range(len(self.classOrder['id'])):
                class_num -= len(self.classOrder['id'][i])
                if class_num == 0:
                    self.now_round = i + 1
                    break
    def reset_by_round(self, now_round):
        """
        use not round to reset imgIds
        """
        
        
        with open(os.path.join('/'.join(self.root_dir.split('/')[:-2]), 'model', 'w_distillation', 'round1' , self.data_split, 'losses.pickle'), 'rb') as f:
            losses = pickle.load(f)
        #Exception
        if now_round == 1:
            raise(ValueError("Now round cannot be 1."))
            
        print('Sample data on Round{}'.format(now_round))
        self.now_round = now_round
        
        if self.method == 'random':
            previous_class_id = []
            future_class_id = []
            for i in range(now_round - 1):
                previous_class_id.extend(self.classOrder['id'][i])
            
            
            for i in range(now_round - 1, len(self.classOrder['id'])):
                future_class_id.extend(self.classOrder['id'][i])
            
            future_imgIds = set(self.cocoHelper.getImgIdFromCats(future_class_id))

            for CID in previous_class_id:
                imgIds = self.cocoHelper.getImgIdFromCats(CID)
                imgIds = set(imgIds) - future_imgIds
                if imgIds == {}:
                    raise(ValueError('Class id:{} contained zero pictures different from now round.'.format(CID)))
                imgIds = imgIds - set(self.image_ids)
                imgIds = list(imgIds)
                
                cur_losses = [losses[img_id][2] for img_id in imgIds]
                
                #cur_losses = [losses[self.coco_labels_inverse[CID]][img_id] for img_id in set(losses[self.coco_labels_inverse[CID]].keys() - future_imgIds)]
                #cur_losses = [losses[self.coco_labels_inverse[CID]][img_id] for img_id in imgIds]
                ids = sorted(range(len(cur_losses)), key=lambda k: cur_losses[k])
                
                #self.image_ids.extend([imgIds[id_] for id_ in ids[:self.per_num]])
                self.image_ids.extend([imgIds[id_] for id_ in ids[len(ids) - self.per_num:]])
                
                # self.image_ids.extend(random.sample(imgIds, self.per_num))
    
                           
    def next_round(self):
        print('rehearsal dataset next round')
        self.now_round += 1
        if self.now_round > len(self.classOrder['id']):
            raise(ValueError("Round{} doesn't exist!.".format(now_round)))
        if self.method == 'random':
            previous_class_id = []
            future_class_id = []
            for i in range(self.now_round - 1):
                previous_class_id.extend(self.classOrder['id'][i])
            for i in range(self.now_round - 1, len(self.classOrder['id'])):
                future_class_id.extend(self.classOrder['id'][i])

            future_imgIds = set(self.cocoHelper.getImgIdFromCats(future_class_id))

            for CID in previous_class_id[int(len(self.image_ids) / self.per_num):]:
                imgIds = self.cocoHelper.getImgIdFromCats(CID)
                imgIds = set(imgIds) - future_imgIds
                if imgIds == {}:
                    raise(ValueError('Class id:{} contained zero pictures different from future round.'.format(CID)))
                imgIds = imgIds - set(self.image_ids)
                imgIds = list(imgIds)
                self.image_ids.extend(random.sample(imgIds, self.per_num))

    def load_classes(self):

        self.coco_labels         = {} # dataloader ID -> origin annotation ID
        self.coco_labels_inverse = {} # origin annotation ID -> dataloader ID

        idx = 0

        for order in self.classOrder['id']:
            for catId in order:
                self.coco_labels[idx] = catId
                self.coco_labels_inverse[catId] = idx
                idx += 1

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        file_name = image_info['file_name']
        path = os.path.join(self.image_path, file_name) #file_name[:-4] mean the image's id
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
            
        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, ann in enumerate(coco_annotations):
#             #When the category doesn't exist in this round, then ignored it 
#             if(ann['category_id'] not in self.seen_class_id):
#                 continue


            # some annotations have basically no width / height, skip them
            if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = ann['bbox']
            annotation[0, 4]  = self.coco_label_to_label(ann['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return int(len(self.prev_imgIds) / self.per_num)
  

class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', dataset = 'coco', transform=None, ):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco      = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        self.dataset = dataset
        self.load_classes()

        if dataset == 'coco':
            self.image_path = os.path.join(self.root_dir, 'images', self.set_name)
        elif dataset == 'voc':
            self.image_path = os.path.join(self.root_dir, 'images')
        with open(os.path.join(self.root_dir, 'path_mapping', set_name + '_path_mapping.pickle'), 'rb') as f:
            self.path_mapping = pickle.load(f)

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes             = {}
        self.coco_labels         = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        file_name = image_info['file_name']
        path = os.path.join(self.image_path, self.path_mapping[int(file_name[:-4])], file_name)
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation        = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4]  = self.coco_label_to_label(a['category_id'])
            annotations       = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]


    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        if self.dataset == 'coco':
            return 80
        elif self.dataset == 'voc':
            return 20


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise(ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise(ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise_from(ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        
        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations     = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 5))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4]  = self.name_to_label(a['class'])
            annotations       = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        #if np.random.rand() < flip_x:
        image, annots = sample['img'], sample['annot']
        image = image[:, ::-1, :]

        rows, cols, channels = image.shape

        x1 = annots[:, 0].copy()
        x2 = annots[:, 2].copy()

        x_tmp = x1.copy()

        annots[:, 0] = cols - x2
        annots[:, 2] = cols - x_tmp

        sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

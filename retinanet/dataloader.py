
from __future__ import print_function, division
import os

import torch
import numpy as np
import random

from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

import skimage.io
import skimage.transform
import skimage.color
import skimage
import pickle

from preprocessing.debug import debug_print, DEBUG_FLAG


class IL_dataset(Dataset):
    """incremental learning dataset."""

    def __init__(self, params,transform=None, start_state=0, use_data_ratio = 1):
        """
        Args:
            train_params: train_params, which manage the training params
            states: IL_states, which manage the incremental learning states
            transform: (callable, optional): Optional transform to be applied on a sample.
            start_state: interger, the start state index
            use_data_ratio: use data ratio, default = 1, which means using all data
        """
    
        self.data_split = params['data_split'] # must be 'train', 'val', 'trainval' or 'test' 
        self.image_path = os.path.join(params['data_path'], 'images')

        self.transform = transform
        self.cur_state = start_state
        self.use_data_ratio = use_data_ratio

        self.states = params.states
        # read annotation and some related data
        self.coco = params.states.coco


        # set annotation seen class
        if self.data_split == "test":
            self.seen_class_id = self.states[self.cur_state]['knowing_class']['id']
        else:
            self.seen_class_id = self.states[self.cur_state]['new_class']['id']

        self.init_classes()
        self.update_imgIds()  #get this state's data
        

    def update_imgIds(self):
        imgIds = self.coco.get_imgs_by_cats(self.seen_class_id)
        if 'test' != self.data_split:
            imgIds = imgIds[:len(imgIds) * self.use_data_ratio]
        self.image_ids = imgIds
      
    def next_state(self):
        debug_print('Dataset Next state!')

        if self.cur_state == len(self.states):
            raise ValueError("Next state doesn't exist.")

        self.cur_state += 1
        self.seen_class_id = self.states[self.cur_state]['new_class']['id']
        self.update_imgIds()
        
    def init_classes(self):
        self.coco_labels         = {} # dataloader ID -> origin annotation ID
        self.coco_labels_inverse = {} # origin annotation ID -> dataloader ID

        for idx, catId in enumerate(self.states[len(self.states) - 1]['knowing_class']['id']):
            self.coco_labels[idx] = catId
            self.coco_labels_inverse[catId] = idx

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
            #When the category doesn't exist in this state, then ignored it 
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

    def num_new_classes(self):
        return self.states[self.cur_state]['num_new_class']
    def num_classes(self):
        return self.states[self.cur_state]['num_knowing_class']

class Replay_dataset(IL_dataset):
    def __init__(self, params, transform=None):
        """
        Args:
            train_params: a Params train_params
            transform: (callable, optional): Optional transform to be applied on a sample.
            start_state: interger, the start state index
            scenario: incremenatl learning state split
            use_data_ratio: use data ratio, default = 1, which means using all data
        """

        super().__init__(params, transform, 1, 1)

        self.per_num = params['sample_num'] #how many picutes each class has 
        self.sample_method = params['sample_method']
        self.cur_state = None
        self.seen_class_id = []
        self.image_ids = []
        if self.sample_method == "large_loss":
            self.large_loss_ckp_path = os.path.join(self.train_params.data_path, 'model', self.train_params.scenario)

                
    def reset_by_imgIds(self, per_num = 1, img_ids = []):
        """reset replay dataset by image ids

            Args:
                per_num: the number of imgs for each classes
                img_ids: the index of images
        """
        self.image_ids = img_ids
        self.per_num = per_num #how many picutes in each class
        self.seen_class_id = []
        if len(self.image_ids) != 0:
            class_num = int(len(self.image_ids) / self.per_num)

            for state in range(len(self.states)):
                if self.states[state]['num_knowing_class'] == class_num:
                    self.cur_state = state + 1
                    self.seen_class_id = self.states[self.cur_state - 1]['knowing_class']['id']
                    return
            raise ValueError("The length of img_ids doesn't meet any state")
        


    def sample_imgs(self, sample_CIDs:list, limit_imgIds:list):
        # read large loss checkpoint
        if self.sample_method == "large_loss":
            
            large_loss_ckp = os.path.join(self.large_loss_ckp_path, "state{}".format(self.cur_state - 1), 'large_losses.pickle')
            with open(large_loss_ckp, 'rb') as f:
                losses = pickle.load(f)

        for CID in sample_CIDs:
            imgIds = self.coco.get_imgs_by_cats(CID)
            imgIds = list(set(imgIds) - limit_imgIds - set(self.image_ids))
            if imgIds == []:
                raise(ValueError('Class id:{} contained zero pictures different from other class in current state.'.format(CID)))

            # large loss sample,(use past loss to sample)
            if self.sample_method == 'large_loss':
                cur_losses = [losses[img_id][2] for img_id in imgIds]
                ids = sorted(range(len(cur_losses)), key=lambda k: cur_losses[k])
                self.image_ids.extend([imgIds[id_] for id_ in ids[len(ids) - self.per_num:]])
            # random sample
            else:
                self.image_ids.extend(random.sample(imgIds, self.per_num))

    def reset_by_state(self, state:int):
        """ use state index to reset imgIds
        """
        # Exception
        if state == 0:
            raise ValueError("Inital state cannot sample picture")
        
        self.cur_state = state
        debug_print('Sample data on state{}'.format(state))
        
        
        self.seen_class_id = self.states[self.cur_state - 1]['knowing_class']['id']

        
        future_CIDs = []
        for i in range(self.cur_state, len(self.states)):
            future_CIDs.extend(self.states[i]['new_class']['id'])

        self.sample_imgs(self.seen_class_id, set(self.coco.get_imgs_by_cats(future_CIDs)))

    def next_state(self):
        """add new state, sample old data
        """
        debug_print('Replay dataloader next state!')
        if self.cur_state == None:
            self.cur_state = 0
        self.cur_state += 1
        if self.cur_state == len(self.states):
            raise(ValueError("State{} doesn't exist in Replay dataloader".format(self.cur_state)))

        sample_CIDs = self.states[self.cur_state - 1]['new_class']['id']
        self.seen_class_id.extend(sample_CIDs)

        # get futrue class ids and futrue imgs
        future_CIDs = []
        for i in range(self.cur_state, len(self.states)):
            future_CIDs.extend(self.states[i]['new_class']['id'])

        self.sample_imgs(self.seen_class_id, set(self.coco.get_imgs_by_cats(future_CIDs)))
        
    # def load_annotations(self, image_index):
    #     # get ground truth annotations
    #     annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
    #     annotations     = np.zeros((0, 5))

    #     # some images appear to miss annotations (like image with id 257034)
    #     if len(annotations_ids) == 0:
    #         return annotations

    #     # parse annotations
    #     coco_annotations = self.coco.loadAnns(annotations_ids)
    #     for idx, ann in enumerate(coco_annotations):
    #         # some annotations have basically no width / height, skip them
    #         if ann['bbox'][2] < 1 or ann['bbox'][3] < 1:
    #             continue

    #         annotation        = np.zeros((1, 5))
    #         annotation[0, :4] = ann['bbox']
    #         annotation[0, 4]  = self.coco_label_to_label(ann['category_id'])
    #         annotations       = np.append(annotations, annotation, axis=0)

    #     # transform from [x, y, w, h] to [x1, y1, x2, y2]
    #     annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
    #     annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

    #     return annotations
  
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

        if np.random.rand() < flip_x:
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

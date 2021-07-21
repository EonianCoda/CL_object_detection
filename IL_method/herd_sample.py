# built-in
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
import pickle
import os
from torchvision import transforms
from retinanet.dataloader import Normalizer, Resizer, Augmenter

THRESOLD = 0.25

# TODO 下面的cat_id指的是dataset_train中的id，並非annotations中的id
def cal_intersection(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)
    intersection = iw * ih
    return intersection

class Feature_resizer(object):
    def __init__(self):
        sizes = [132, 66, 33, 17, 9]
        self.f = []
        for i in sizes:
            size = (i,i)
            self.f.append(transforms.Resize(size))
    def __call__(self, x):
        for idx, feature in enumerate(x):
            x[idx] = self.f[idx](feature).flatten()
        return torch.cat(x)
    

class Herd_sampler(object):
    def __init__(self,il_trainer):
        self.il_trainer = il_trainer
        self.ratio_thresold = THRESOLD

        path = os.path.join(self.il_trainer.params['ckp_path'], 'state{}'.format(self.il_trainer.cur_state))

        # self.sample_file_name = os.path.join(path, 'sample{}.pickle'.format(self.ratio_thresold))
        self.mean_file_name = os.path.join(path, 'mean_feature{}.pickle'.format(self.ratio_thresold))
        self.scores_file_name = os.path.join(path, 'classified_scores{}.pickle'.format(self.ratio_thresold))

        self.feature_resizer = Feature_resizer()
        self.examplar_dict = defaultdict(list)
        self.examplar_list = list()


    def update_examplar(self,examplar_dict, examplar_list):
        for cat_id, img_ids in examplar_dict.items():
            self.examplar_dict[cat_id].extend(img_ids)
        self.examplar_list.extend(examplar_list)

    def sample(self, per_num):
        dataset = self.il_trainer.dataset_train
        self.per_num = int(per_num)

        # # if the exam
        # if os.path.isfile(self.sample_file_name):
        #     with open(self.sample_file_name, 'rb') as f:
        #         examplar_dict, examplar_list = pickle.load(f)
        #         self.examplar_dict, self.examplar_list = examplar_dict, examplar_list


        # assign class for images
        classified_ratios = self._cal_foreground_ratio()
        classified_imgs = self._discard_low_ratio(classified_ratios, self.ratio_thresold) # cat_id => img_ids
        reverse_classified_imgs = defaultdict(list) # img_id => cat_id


        
        for img_id in dataset.image_ids:
            for cat_id, img_ids in classified_imgs.items():
                if img_id in img_ids:
                    reverse_classified_imgs[img_id].append(cat_id)

        dataset.transform = transforms.Compose([Normalizer(),Resizer()])
        # calculate the mean feature for each category
        if not os.path.isfile(self.mean_file_name):
            mean_features = self._cal_mean_feature(classified_imgs, reverse_classified_imgs)

            for cat_id in mean_features.keys():
                mean_features[cat_id] = mean_features[cat_id].cpu()
            with open(self.mean_file_name,'wb') as f:
                pickle.dump(mean_features, f)
        else:
            with open(self.mean_file_name,'rb') as f:
                mean_features = pickle.load(f)
        
        if not os.path.isfile(self.scores_file_name):
            for cat_id in mean_features.keys():
                mean_features[cat_id] = mean_features[cat_id].cuda()

            scores = self._cal_difference(mean_features, classified_imgs, reverse_classified_imgs)
            with open(self.scores_file_name,'wb') as f:
                pickle.dump(scores, f)
        else:
            with open(self.scores_file_name,'rb') as f:
                scores = pickle.load(f)

        examplar_dict, examplar_list = self._sample_by_scores(scores, per_num)
        self.update_examplar(examplar_dict, examplar_list)
        # self.save_examplar()

        dataset.transform = transforms.Compose([Normalizer(),Augmenter(), Resizer()])
        # destroy
        del mean_features
    def save_examplar(self):
        with open(self.sample_file_name, 'wb') as f:
            pickle.dump( (self.examplar_dict, self.examplar_list), f)

    def get_named_examplar(self):
        dataset_train = self.il_trainer.dataset_train
        named_examplar = {}
        for cat_id, examplar in self.examplar_dict.items():
            cat_id = dataset_train.label_to_coco_label(cat_id)
            cat_name = dataset_train.coco.catId_to_name(cat_id)[0]
            named_examplar[cat_name] = examplar
        return named_examplar

    def save_examplar_image(self, saved=True):
        """generate the image which contains all examplars
            Args:
                saved: whether save the figute or return, default=True, False mean return the figure
        """
        img_path = self.il_trainer.dataset_train.image_path
        num_classes = int(len(self.examplar_list) / self.per_num)
        fig = plt.figure(figsize=(4*self.per_num,3.5*num_classes), constrained_layout=True)
        gs = fig.add_gridspec(num_classes, self.per_num)

        row = 0
        named_examplar = self.get_named_examplar()
        for cat_name, examplar in named_examplar.items():
            for idx, img_id in enumerate(examplar):
                ax = fig.add_subplot(gs[row, idx])
                im = cv2.imread(os.path.join(img_path, "{:06d}".format(img_id) +'.jpg'))
                ax.set_title(cat_name)
                ax.imshow(im)
            row += 1
        
        if saved:
            path = os.path.join(self.il_trainer.params['ckp_path'], 'state{}'.format(self.il_trainer.cur_state))
            file_name = os.path.join(path, "examplar.png")
            plt.savefig(file_name)
        else:
            return fig 

    def _sample_by_scores(self, scores, per_num):

        examplar_dict = defaultdict(list)
        examplar_list = []

        for cat_id, img_score in scores.items():
            sorted_img_ids = sorted(img_score.keys(), key=lambda k: img_score[k])
            i = 0
            for img_id in sorted_img_ids:
                if img_id not in examplar_list:
                    examplar_dict[cat_id].append(img_id)
                    examplar_list.append(img_id)
                    i += 1
                    if i == per_num:
                        break
        return examplar_dict, examplar_list
    
    def _cal_difference(self, mean_features, classified_imgs, reverse_classified_imgs):
        def distance(mean, x):
            return float(torch.norm(mean - x))


        dataset = self.il_trainer.dataset_train
        model = self.il_trainer.model

        all_classes = set(self.il_trainer.params.states[-1]['knowing_class']['id'])
        cur_knowing_classes = set(self.il_trainer.params.states[self.il_trainer.cur_state]['knowing_class']['id'])
        future_cat_ids = list(all_classes - cur_knowing_classes)
        future_img_ids = dataset.coco.get_imgs_by_cats(future_cat_ids)

        scores = {cat_id:dict() for cat_id in classified_imgs.keys()}

        for idx in range(len(dataset)):
            img_id = dataset.image_ids[idx]
            # if image isn't assigned a class or contains the foreground for future class, then ignore
            if not reverse_classified_imgs.get(img_id) or img_id in future_img_ids:
                continue

            data = dataset[idx]
            features = self._cal_feature(model, data)
            for cat_id in reverse_classified_imgs[img_id]:
                scores[cat_id][img_id] = distance(features, mean_features[cat_id])
        return scores
        
    def _discard_low_ratio(self, ratios, thresold):
        classified_imgs = defaultdict(list)
        for img_id, ratio in ratios.items():
            for cat_id, foreground_ratio in ratio.items():
                if foreground_ratio >= thresold:
                    classified_imgs[cat_id].append(img_id)
        return classified_imgs

    def _cal_foreground_ratio(self):
        """calculate the area of the foreground / the area of image
        """
        img_classified_ratio = {}
        coco = self.il_trainer.dataset_train.coco
        i = 1
        for idx in range(len(self.il_trainer.dataset_train)):
            classified_ratio = defaultdict(float)

            #calculate the area of the image
            img_id = self.il_trainer.dataset_train.image_ids[idx]
            img_info = coco.loadImgs(ids=img_id)[0]
            img_area = img_info['width'] * img_info['height']

            #calculate the area of the foreground
            annots = self.il_trainer.dataset_train.load_annotations(idx)
            annots = annots[annots[:,-1] != -1]
            labels = annots[:,-1].astype('int') # get lables for annotations 
            
            classified_annots = defaultdict(list)
            for i, cat_id in enumerate(labels):
                classified_annots[cat_id].append(annots[i,:])

            for cat_id, boxes in classified_annots.items():
                boxes = torch.tensor(np.array(boxes))
                intersection = cal_intersection(boxes,boxes)
                foreground_area = 0.0
                for row in range(intersection.shape[0]):
                    for col in range(row + 1):
                        if row == col:
                            foreground_area += float(intersection[row][col])
                        else:
                            foreground_area -= float(intersection[row][col])
                classified_ratio[cat_id] = foreground_area / img_area
            img_classified_ratio[img_id] = classified_ratio
        return img_classified_ratio
        
    def _cal_feature(self, model, data):
        with torch.no_grad():    
            features = model.forward_feature(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            features = [feature.squeeze() for feature in features]
            features = self.feature_resizer(features)
        return features

    def _cal_mean_feature(self, classified_imgs, reverse_classified_imgs):
        """calculate the mean feature for each category
        """
        
        mean_features = {}
        dataset = self.il_trainer.dataset_train
        model = self.il_trainer.model
        
        for i in range(len(dataset)):
            img_id = dataset.image_ids[i]

            # The image isn't assigned a category
            if not reverse_classified_imgs.get(img_id):
                continue
            
            data = dataset[i]
            features = self._cal_feature(model, data)
            for cat_id in reverse_classified_imgs[img_id]:
                if mean_features.get(cat_id) == None:
                    mean_features[cat_id] = features.clone() / len(classified_imgs[cat_id])
                else:
                    mean_features[cat_id] += features / len(classified_imgs[cat_id])

        return mean_features

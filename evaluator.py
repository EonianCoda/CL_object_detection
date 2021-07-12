# built-in
import argparse
from tqdm import tqdm
import os
import json
from collections import defaultdict
import numpy as np
# torch
import torch
from torchvision import transforms
# pycocotools
from pycocotools.cocoeval import COCOeval

# retinanet
from retinanet.dataloader import IL_dataset, Resizer, Normalizer
from retinanet.model import create_retinanet
from preprocessing.params import Params, create_dir


class Evaluator(Params):
    def __init__(self, parser:argparse):
        super().__init__(parser, "test")
        self.model = None
        self.init_dataset()

    # def output_evaluation_file(self, precsion:dict, recall:dict):
    #     cat_names = [name for name in precsion.keys()]
        
    def get_model(self, epoch=None):
        """
            Args:
                state: int. the index of state
                epoch: int, -1 mean auto search the max epoch in this state , None mean not readcheckpoint default = None , 
        """
        model = create_retinanet(self['depth'], self.states[self['state']]['num_knowing_class'])
        if epoch == None:
            return model
        self.load_model(self['state'], epoch, model)
        return model

    def do_evaluation(self, epoch:int, ignore_other_img=False):
        """do model predict
        Args:
            ignore_other_img: whether ignore the img not containing own category
        """
        # load results in COCO evaluation tool

        coco_true = self.dataset.coco
        coco_pred = coco_true.loadRes(self.get_result_path(epoch))
        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = self.dataset.image_ids
        
        precision_result = defaultdict()
        recall_result = defaultdict()
        for class_id in self.dataset.seen_class_id:
            class_name = self.dataset.coco.catId_to_name(class_id)[0]
            print('Evaluate {}:'.format(class_name))
            coco_eval.params.catIds = [class_id] #dataset.seen_class_id
            if ignore_other_img:
                coco_eval.params.imgIds = self.dataset.coco.get_imgs_by_cats(class_id)
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            precision_result[class_name] = coco_eval.stats[1]
            recall_result[class_name] = coco_eval.stats[8]
            
        if len(self.dataset.seen_class_id) > 1:
            print("Precision:")
            for name, ap in sorted(precision_result.items()):
                print('{:<12} = {:0.2f}'.format(name, ap))
                
            print("Recall:")
            for name, ap in sorted(recall_result.items()):
                print('{:<12} = {:0.2f}'.format(name, ap))
            
            print("-"*50)
            print('{:<12} = {:0.2f}'.format('MAP', np.mean([v for v in precision_result.values()])))
            print('{:<12} = {:0.2f}'.format('Average Recall', np.mean([v for v in recall_result.values()])))
            print("Precision:")
            for name, ap in sorted(precision_result.items()):
                print('{:0.2f}'.format(ap))
                
            print("Recall:")
            for name, ap in sorted(recall_result.items()):
                print('{:0.2f}'.format(ap))
        
    def init_dataset(self):
        self.dataset = IL_dataset(self,
                                transform=transforms.Compose([Normalizer(), Resizer()]),
                                start_state=self['state'])
    def get_result_path(self, epoch:int):
        # write output
        root_dir = self['root_dir']
        file_path = os.path.join(root_dir, 'val_result')
        create_dir(file_path)
        file_path = os.path.join(file_path, self['scenario'])
        create_dir(file_path)
        file_path = os.path.join(file_path, 'state{}'.format(self['state']))
        create_dir(file_path)

        file_name = '{}_results_epoch{}.json'.format(self['dataset'], epoch)
        return os.path.join(file_path, file_name)

    def do_predict(self, epoch=None, pbar=None):
        """do prediction
        
            Args:
                epoch: int, default=None
        """
        if epoch == None:
            raise ValueError("Epoch cannot be None")

        model = self.get_model(epoch)
        model = model.cuda()
        model.training = False
        model.eval()
        model.freeze_bn()
        with torch.no_grad():
            # start collecting results
            results = []
            image_ids = []


            if pbar:
                iterator_ = tqdm(range(len(self.dataset)),position=0, leave=True)
            else:
                iterator_ = range(len(self.dataset))

            for index in iterator_:

                data = self.dataset[index]
                scale = data['scale']

                # run network
                scores, labels, boxes = model.predict(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))

            
                scores = scores.cpu()
                labels = labels.cpu()
                boxes  = boxes.cpu()

                # correct boxes for image scale
                boxes /= scale

                if boxes.shape[0] > 0:
                    # change to (x, y, w, h) (MS COCO standard)
                    boxes[:, 2] -= boxes[:, 0]
                    boxes[:, 3] -= boxes[:, 1]

                    # compute predicted labels and scores
                    #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    for box_id in range(boxes.shape[0]):
                        score = float(scores[box_id])
                        label = int(labels[box_id])
                        box = boxes[box_id, :]

                        # scores are sorted, so we can break
                        if score < self['threshold']:
                            continue

                        # append detection for each positively labeled class
                        image_result = {
                            'image_id'    : self.dataset.image_ids[index],
                            'category_id' : self.dataset.label_to_coco_label(label),
                            'score'       : float(score),
                            'bbox'        : box.tolist(),
                        }

                        # append detection to results
                        results.append(image_result)

                # append image to list of processed images
                image_ids.append(self.dataset.image_ids[index])
                
                if pbar:
                    pbar.update(1)

            if not len(results):
                return


            file_path = self.get_result_path(epoch)
            json.dump(results, open(os.path.join(file_path), 'w') ,indent=4)
            print("Prediction Foreground num = {}".format(len(results)))

        model.cpu()
        del model
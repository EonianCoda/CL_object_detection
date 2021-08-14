# built-in
from IL_method.bic import Bic_Evaluator
import argparse
import copy
from tqdm import tqdm
import os
import json
from collections import defaultdict
import numpy as np
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime 
# torch
import torch
from torchvision import transforms
# pycocotools
from pycocotools.cocoeval import COCOeval

# retinanet
from retinanet.dataloader import IL_dataset, Resizer, Normalizer
from retinanet.model import create_retinanet
from preprocessing.params import Params, create_dir

DEFAULT_RESULT = {'precision':[], 'recall':[],'pred_num':0,'real_num':0}


class Evaluator(Params):
    def __init__(self, parser:argparse):
        if parser['eval_on_train']:
            super().__init__(parser)
        else:
            super().__init__(parser, "test")
        self.model = None
        self.init_dataset()
        self.results = {}
        self.collect_result = self['output_csv']
        if self['new_folder']:
            if self['specific_folder'] == "None":
                self.new_folder_name = datetime.now().strftime("%Y-%m-%d-%H-%M")
            else:
                self.new_folder_name = self['specific_folder']
    

    def get_tensorbord_info(self):
        """generate the information which will be recored into tensorboard
        """
        results = dict()
        ap_declines = defaultdict(list)
        recall_declines = defaultdict(list)
        file_path = os.path.join(self['root_dir'], 'val_result')
        with open(os.path.join(file_path, 'upper_bound.pickle'), 'rb') as f:
            upper_bound = pickle.load(f)

        cat_names = self.states[self['state']]['knowing_class']['name']
        epochs = [epoch for epoch in self.results.keys()]
        epochs.sort()
        cat_num = len(self.dataset.seen_class_id)

        # result
        for idx in range(cat_num):
            cat_name = cat_names[idx]
            upper_bound_ap = upper_bound[cat_name]['ap']
            upper_bound_recall = upper_bound[cat_name]['recall']
            for epoch in epochs:
                ap = self.results[epoch]['precision'][idx]
                recall = self.results[epoch]['recall'][idx]
                ap_declines[epoch].append(upper_bound_ap - ap)
                recall_declines[epoch].append(upper_bound_recall - recall)

        # sum of old Decline
        old_class_num = len(self.states[self['state'] - 1]['knowing_class']['id'])
        for epoch in epochs:
            results[epoch] = dict()
            results[epoch]['sum_ap_decline'] = sum(ap_declines[epoch][:old_class_num]) * 100
            results[epoch]['sum_recall_decline'] = sum(recall_declines[epoch][:old_class_num]) * 100

            num_classes = len(self.results[epoch]['precision'])
            num_new_classes = num_classes - old_class_num
            results[epoch]['new_class_ap'] = sum(self.results[epoch]['precision'][old_class_num:]) / num_new_classes
            results[epoch]['new_class_recall'] = sum(self.results[epoch]['recall'][old_class_num:]) / num_new_classes
            results[epoch]['pred_ratio'] = self.results[epoch]['pred_num'] / self.results[epoch]['real_num']
        return results

    def output_csv_file(self):
        if self.results == {}:
            return

        ap_declines = defaultdict(list)
        recall_declines = defaultdict(list)
        file_path = os.path.join(self['root_dir'], 'val_result')
        with open(os.path.join(file_path, 'upper_bound.pickle'), 'rb') as f:
            upper_bound = pickle.load(f)

        cat_names = self.states[self['state']]['knowing_class']['name']
        epochs = [epoch for epoch in self.results.keys()]
        epochs.sort()
        cat_num = len(self.dataset.seen_class_id)

        lines = []
        line = ''
        # Description
        line = 'Epoch'
        for epoch in epochs:
            line += ',{}'.format(epoch) * 4
        lines.append(line)
        line = ''
        for _ in epochs:
            line += ',AP,Recall,AP_decline, Recall_decline'
        lines.append(line)
        # result
        for idx in range(cat_num):
            cat_name = cat_names[idx]
            line = cat_name
            upper_bound_ap = upper_bound[cat_name]['ap']
            upper_bound_recall = upper_bound[cat_name]['recall']
            for epoch in epochs:
                ap = self.results[epoch]['precision'][idx]
                recall = self.results[epoch]['recall'][idx]
                ap_declines[epoch].append(upper_bound_ap - ap)
                recall_declines[epoch].append(upper_bound_recall - recall)
                line += ',{},{},{:.1f}%,{:.1f}%'.format(ap, 
                                              recall,
                                              ap_declines[epoch][-1]*100,
                                              recall_declines[epoch][-1]*100)
            lines.append(line)
        # Mean
        line = 'Mean'
        for epoch in epochs:
            mean_ap = np.mean(self.results[epoch]['precision'])
            mean_recall = np.mean(self.results[epoch]['recall'])
            upper_bound_mean_ap = upper_bound['mean']['ap']
            upper_bound_mean_recall = upper_bound['mean']['recall']
            line += ',{},{},{:.1f}%,{:.1f}%'.format(mean_ap, 
                                                    mean_recall,
                                                    (upper_bound_mean_ap - mean_ap)*100,
                                                    (upper_bound_mean_recall - mean_recall)*100)
        lines.append(line)

        # sum of old Decline
        old_class_num = len(self.states[self['state'] - 1]['knowing_class']['id'])
        line = 'Sum_decline'
        for epoch in epochs:
            line += ',,,{:.1f}%,{:.1f}%'.format(sum(ap_declines[epoch][:old_class_num]) * 100, sum(recall_declines[epoch][:old_class_num]) * 100)
        lines.append(line)
        
        # pred and real num
        line = 'Pred num'
        for epoch in epochs:
            line += ',{},,,'.format(self.results[epoch]['pred_num'])
        lines.append(line)
        line = 'Pred ratio'
        for epoch in epochs:
            line += ',{:.1f},,{:.1f},'.format(self.results[epoch]['pred_num'] / self.results[epoch]['real_num'], upper_bound['pred_ratio'])
        lines.append(line)


        lines = '\n'.join(lines)
        file_name = 'val_result_' + '_'.join([str(epoch) for epoch in epochs]) + '.csv'
        file_path = self.get_result_path(-1)
        with open(os.path.join(file_path, file_name), 'w') as f:
            f.write(lines)

    def evaluation_check(self, epochs):
        """Before donig evaluation, check if the checkpoint file exists
            Args:
                epcohs: int or list, containing the indexs of epochs 
        """
        if isinstance(epochs, int):
            epochs = [epochs]

        for epoch in epochs:
            ckp_file = self.get_ckp_path(self['state'], epoch)
            if not os.path.isfile(ckp_file):
                raise ValueError("{} is not found!".format(ckp_file))

    def validation_check(self, epochs):
        """Before doing validation, check if the result file exists
            Args:
                epcohs: int or list, containing the indexs of epochs 
        """
        if isinstance(epochs, int):
            epochs = [epochs]

        for epoch in epochs:
            pred_file = self.get_result_path(epoch)
            if not os.path.isfile(pred_file):
                raise ValueError("{} is not found!".format(pred_file))

    def do_evaluation(self, epoch:int, ignore_other_img=False):
        """do model predict
        Args:
            ignore_other_img: whether ignore the img not containing own category
        """
        # load results in COCO evaluation tool

        pred_file = self.get_result_path(epoch)
        if not os.path.isfile(pred_file):
            raise ValueError("{} not found!".format(pred_file))

        coco_true = self.dataset.coco
        coco_pred = coco_true.loadRes(pred_file)

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
        
        precision_result = sorted(precision_result.items())
        recall_result = sorted(recall_result.items())
        if self.collect_result:
            empty_result = copy.deepcopy(DEFAULT_RESULT)
            for idx in range(len(precision_result)):
                empty_result['precision'].append(precision_result[idx][1])
                empty_result['recall'].append(recall_result[idx][1])
            
            empty_result['pred_num'] = len(coco_pred.getAnnIds())
            empty_result['real_num'] = len(coco_true.getAnnIds(catIds=self.dataset.seen_class_id))
            self.results[epoch] = empty_result

    def init_dataset(self):
        self.dataset = IL_dataset(self,
                                transform=transforms.Compose([Normalizer(), Resizer()]),
                                start_state=self['state'])
    def get_result_path(self, epoch:int):
        """
            Args:
                epoch: int, if epoch == -1, then only return  directory path
        """
        # write output
        root_dir = self['root_dir']
        file_path = os.path.join(root_dir, 'val_result')
        create_dir(file_path)
        file_path = os.path.join(file_path, self['scenario'])
        create_dir(file_path)
        file_path = os.path.join(file_path, 'state{}'.format(self['state']))
        create_dir(file_path)
        if self['new_folder']:
            file_path = os.path.join(file_path, self.new_folder_name)
            create_dir(file_path)

        if epoch != -1:
            file_name = '{}_results_epoch{}.json'.format(self['dataset'], epoch)
            return os.path.join(file_path, file_name)
        else:
            return file_path

    def do_predict(self, epoch=None, pbar=None, indexs=None):
        """do prediction
        
            Args:
                epoch: int, default=None
        """
        if epoch == None:
            raise ValueError("Epoch cannot be None")
        if indexs != None:
            just_return = True
        else:
            just_return = False
        model = self.get_model(self['state'], epoch)
        model = model.cuda()
        model.training = False
        model.eval()
        model.freeze_bn()
        if self['bic']:
            bic_evaluator = Bic_Evaluator(self, self['state'])
            bic_file = os.path.join(self['ckp_path'], 'state{}'.format(self['state']), 'bic_{}.pt'.format(epoch))
            bic_evaluator.load_ckp(bic_file)

        with torch.no_grad():
            # start collecting results
            results = []
            image_ids = []

            if indexs == None:
                indexs = range(len(self.dataset))
            for index in indexs:

                data = self.dataset[index]
                scale = data['scale']

                # run network
                if self['bic']:
                    scores, labels, boxes = model.predict(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0), bic=bic_evaluator)
                else:
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

            if not just_return:
                file_path = self.get_result_path(epoch)
                json.dump(results, open(os.path.join(file_path), 'w') ,indent=4)
                print("Prediction Foreground num = {}".format(len(results)))

        model.cpu()
        del model
        if just_return:
            return results


def multi_evaluation(evaluator:Evaluator, epochs:list):
    evaluator.evaluation_check(epochs)
    def single_evaluation(epoch:int, pbar=None):
        evaluator.do_predict(epoch, pbar)
        evaluator.do_evaluation(epoch)

    def multi_split_evaluation(epoch:int, pbar, split=2):
        def single_just_evaluation(epoch, pbar, indexs):
            return evaluator.do_predict(epoch, pbar, indexs)

        indexs = [i for i in range(len(evaluator.dataset))]
        part_len = int(len(indexs) / split)
        eval_indexs = []
        for i in range(split-1):
            eval_indexs.append(indexs[i*part_len:(i+1)*part_len])
        eval_indexs.append(indexs[(split - 1)*part_len:])
        results = []
        with ThreadPoolExecutor(max_workers=split) as ex:
            futures = [ex.submit(single_just_evaluation, epoch, pbar, idxs) for idxs in eval_indexs]
            for future in as_completed(futures):                
                results.extend(future.result())
        
        file_path = evaluator.get_result_path(epoch)
        json.dump(results, open(os.path.join(file_path), 'w') ,indent=4)
        evaluator.do_evaluation(epoch)



    with tqdm(total=len(evaluator.dataset) * len(epochs),position=0, leave=True) as pbar:
        with ThreadPoolExecutor(max_workers=len(epochs)) as ex:
            if len(epochs) > 1:
                for epoch in epochs:
                    ex.submit(single_evaluation, epoch, pbar)
            else:
                multi_split_evaluation(epochs[0], pbar, split=5)



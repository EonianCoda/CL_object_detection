import os
import pickle
from preprocessing.params import create_dir
from retinanet.dataloader import IL_dataset
import torchvision
from torchvision import transforms
import torch

from retinanet.dataloader import Resizer,  Normalizer
from retinanet.losses import calc_iou

DEFAULT_SCORE_THRESOLD = 0.7
DEFAULT_IOU_THRESOLD = 0.35

class Labeler():
    def __init__(self, model, params, score_thresold = DEFAULT_SCORE_THRESOLD, IOU_thresold=DEFAULT_IOU_THRESOLD):
        self.regressBoxes = model.regressBoxes
        self.clipBoxes = model.clipBoxes
        self.model = model
        self.params = params
        self.score_thresold = score_thresold
        self.IOU_thresold = IOU_thresold

    def label_data(self, state:int):
        dataset = IL_dataset(self.params,
                            transform=transforms.Compose([Normalizer(), Resizer()]),
                            start_state=state)

        path = os.path.join(self.params['ckp_path'], 'state{}'.format(state))
        file_name = "persuado_label_{}_{}.pickle".format(self.score_thresold, self.IOU_thresold)
        create_dir(path)

        # if the file exists
        if os.path.isfile(os.path.join(path, file_name)):
            with open(os.path.join(path, file_name), 'rb') as f:
                return pickle.load(f)
    
        persuado_annots = {}
        for iter_num, data in enumerate(dataset):
            img_id = dataset.image_ids[iter_num]
            with torch.no_grad():
                img_batch = data['img'].permute(2, 0, 1).float().unsqueeze(dim=0).cuda()
                annotations = data['annot'].unsqueeze(dim=0).cuda()
                classifications, regressions, anchors = self.model(img_batch, 
                                                                    return_feat=False, 
                                                                    return_anchor=True, 
                                                                    enable_act=True)
                prediction_scores, preidction_boxes, prediction_targets = self.predict(img_batch, classifications, regressions, anchors)
            
            scale = data['scale']

            # get the anchor which its scores are large than thresold 
            if len(preidction_boxes) != 0:
                mask = (prediction_scores > self.score_thresold)
                preidction_boxes = preidction_boxes[mask] / scale
                prediction_scores = prediction_scores[mask]
                prediction_targets = prediction_targets[mask]

            # get the anchor which its IOU are less than any other groud truth label
            if len(preidction_boxes) != 0:
                # get groud truth boxes and labels
                scale = data['scale']
                annotation = annotations[0,...]
                annotation = annotation[annotation[..., -1] != -1]
                gd_boxes = annotation[..., :4]
                gd_boxes /= scale

                IOU_result = calc_iou(preidction_boxes, gd_boxes)
                max_IOU , _ = IOU_result.max(dim=1)

                mask = max_IOU < self.IOU_thresold

                preidction_boxes = preidction_boxes[mask]
                prediction_scores = prediction_scores[mask]
                prediction_targets = prediction_targets[mask]


            # get persudo label
            results = []
            if preidction_boxes.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                preidction_boxes[:, 2] -= preidction_boxes[:, 0]
                preidction_boxes[:, 3] -= preidction_boxes[:, 1]

                for i in range(preidction_boxes.shape[0]):
                    result = {'category_id' : dataset.label_to_coco_label(int(prediction_targets[i])),
                                'score' : float(prediction_scores[i]),
                                'bbox'  : preidction_boxes[i].tolist(),
                                }
                    results.append(result)
            persuado_annots[img_id] = results

        # store result
        with open(os.path.join(path, file_name), 'rb') as f:
            pickle.dump(persuado_annots, f)

        return persuado_annots

    def predict(self, img_batch, classifications, regressions, anchors):
        transformed_anchors = self.regressBoxes(anchors, regressions)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
        
        # for batch size 1
        j = 0
        classification = classifications[j, :, :] # shape = (num_anchors, class_num)
        transformed_anchor = transformed_anchors[j,...]

        prediction_pos_mask = (classification > 0.05).any(dim=1)

        scores, target_ids = torch.max(classification, dim=1)

        scores = scores[prediction_pos_mask]
        target_cls_ids = target_ids[prediction_pos_mask]
        transformed_anchor = transformed_anchor[prediction_pos_mask]

        final_anchor_ids = torchvision.ops.batched_nms(transformed_anchor, scores, target_cls_ids, 0.5)
        
        if len(final_anchor_ids) == 0:
            prediction_scores = torch.tensor([])
            preidction_boxes = torch.tensor([])
            prediction_targets = torch.tensor([])
        else:
            prediction_scores = scores[final_anchor_ids]
            preidction_boxes = transformed_anchor[final_anchor_ids]
            prediction_targets = target_cls_ids[final_anchor_ids]

        return prediction_scores, preidction_boxes, prediction_targets


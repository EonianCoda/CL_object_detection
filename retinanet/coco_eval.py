from pycocotools.cocoeval import COCOeval
import json
import torch
from tqdm import tqdm
import os
from collections import defaultdict
import numpy as np
def checkDir(path):
    """check whether directory exists or not.If not, then create it 
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def evaluate_coco(dataset, model, root_path, method, now_round, epoch, threshold=0.05):
    
    model.eval()
    
    with torch.no_grad():

        # start collecting results
        results = []
        image_ids = []

        for index in tqdm(range(len(dataset))):
            data = dataset[index]
            scale = data['scale']

            # run network
            if torch.cuda.is_available():
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = model(data['img'].permute(2, 0, 1).float().unsqueeze(dim=0))
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
                    if score < threshold:
                        continue

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            #print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        
        path = os.path.join(root_path, 'valResult', 'Voc', method)
        checkDir(path)
        path = os.path.join(path, "round{}".format(now_round))
        checkDir(path)
      
        json.dump(results, open(os.path.join(path , '{}_bbox_results_{}_for{}epoch.json'.format(dataset.set_name, now_round, epoch)), 'w') ,indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes(os.path.join(path, '{}_bbox_results_{}_for{}epoch.json'.format(dataset.set_name, now_round, epoch)))
        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        
        precision_result = defaultdict()
        recall_result = defaultdict()
        for class_id in dataset.seen_class_id:
            class_name = dataset.cocoHelper.catIdToName(class_id)[0]
            print('Evaluate {}:'.format(class_name))
            coco_eval.params.catIds = [class_id] #dataset.seen_class_id
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            precision_result[class_name] = coco_eval.stats[1]
            recall_result[class_name] = coco_eval.stats[8]
            
        if len(dataset.seen_class_id) > 1:
            print("Precision:")
            for name, ap in sorted(precision_result.items()):
                print('{:<12} = {:0.2f}'.format(name, ap))
                
            print("Recall:")
            for name, ap in sorted(recall_result.items()):
                print('{:<12} = {:0.2f}'.format(name, ap))
            
            print("------------------------------------------")
            print('{:<12} = {:0.2f}'.format('MAP', np.mean([v for v in precision_result.values()])))
            print('{:<12} = {:0.2f}'.format('Average Recall', np.mean([v for v in recall_result.values()])))
            print("Precision:")
            for name, ap in sorted(precision_result.items()):
                print('{:0.2f}'.format(ap))
            print("Recall:")
            for name, ap in sorted(recall_result.items()):
                print('{:0.2f}'.format(ap))
            
        model.train()
        
        return

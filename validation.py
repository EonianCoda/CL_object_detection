# built-in
import argparse
import tqdm
import os
import json
from collections import defaultdict
import numpy as np
# torch
import torch
from torchvision import transforms
from retinanet import model

# retinanet
from retinanet.dataloader import IL_dataset, Resizer, Normalizer
from retinanet.model import create_retinanet
from preprocessing.params import Params, create_dir
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

ROOT_DIR = "/home/deeplab307/Documents/Anaconda/Shiang/IL/"
DEFAULT_THRESHOLD = 0.05
DEFAULT_DEPTH = 50

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_val_parser(args=None):
    parser = argparse.ArgumentParser()
    # must set params
    
    parser.add_argument('--dataset', help='Dataset name, must contain name and years, for instance: voc2007,voc2012', default='voc2007')
    parser.add_argument('--epoch', help='the index of validation epochs', type=int)
    parser.add_argument('--state', type=int)
    parser.add_argument('--scenario', help='the scenario of states, must be "20", "19 1", "10 10", "15 1", "15 1 1 1 1"', type=int, nargs="+", default=[20])
    parser.add_argument('--threshold', help='the threshold for prediction default=0.05', type=float, default=DEFAULT_THRESHOLD)

    # always fixed
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=DEFAULT_DEPTH)
    parser.add_argument('--root_dir', help='the root dir for training', default=ROOT_DIR)
    parser = vars(parser.parse_args(args))
    # set for origin Parmas, otherwise it will have error
    parser['warm_stage'] = 0
    parser['shuffle_class'] = False
    return parser


class Evaluator(Params):
    def __init__(self, parser:argparse):
        super().__init__(parser, "test")
        self.model = None
        self.init_dataset()
        self.init_model()
    def do_evalation(self, ignore_other_img=False):
        """do model predict
        Args:
            ignore_other_img: whether ignore the img not containing own category
        """
        # load results in COCO evaluation tool

        coco_true = self.dataset.coco
        coco_pred = coco_true.loadRes(self.get_result_path())
        # run COCO evaluation
        coco_eval = COCO(coco_true, coco_pred, 'bbox')
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
            
            print("------------------------------------------")
            print('{:<12} = {:0.2f}'.format('MAP', np.mean([v for v in precision_result.values()])))
            print('{:<12} = {:0.2f}'.format('Average Recall', np.mean([v for v in recall_result.values()])))
            print("Precision:")
            for name, ap in sorted(precision_result.items()):
                print('{:0.2f}'.format(ap))
            print("Recall:")
            for name, ap in sorted(recall_result.items()):
                print('{:0.2f}'.format(ap))
        
    def init_model(self):
        # if self.model != None:
        #     self.model.cpu()
        #     del self.model
        self.model = create_retinanet(self['depth'], self.states[self['state']]['num_knowing_class'])
        self.model = self.model.cuda()
        self.load_model(self['state'], self['epoch'], self.model)
        self.model.training = False
        self.model.eval()
        self.model.freeze_bn()

    def init_dataset(self):
        self.dataset = IL_dataset(self,
                                transform=transforms.Compose([Normalizer(), Resizer()]),
                                start_state=self['state'])
    def get_result_path(self):
        # write output
        root_dir = self['root_dir']
        file_path = os.path.join(root_dir, 'val_result')
        create_dir(file_path)
        file_path = os.path.join(file_path, self['scenario'])
        create_dir(file_path)
        file_path = os.path.join(file_path, 'state{}'.format(self['state']))
        create_dir(file_path)

        file_name = '{}_results_epoch{}.json'.format(self['dataset'], self['epoch'])
        return os.path.join(file_path, file_name)

    def do_predict(self, state=None, epoch=None):
        """do prediction
        """
        self.model.eval()
        with torch.no_grad():
            # start collecting results
            results = []
            image_ids = []

            for index in tqdm(range(len(self.dataset))):
                data = self.dataset[index]
                scale = data['scale']

                # run network
                scores, labels, boxes = self.model.predict(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))

            
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
                        if score < self['thresold']:
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


            if not len(results):
                return


            file_path = self.get_result_path()
            json.dump(results, open(os.path.join(file_path), 'w') ,indent=4)
            print("Prediction Foreground num = {}".format(len(results)))

def main(args=None):
    parser = get_val_parser(args)
    evaluator = Evaluator(parser)
    evaluator.do_predict()
    evaluator.do_evalation()

if __name__ == '__main__':
    assert torch.__version__.split('.')[0] == '1'
    if not torch.cuda.is_available():
        print("CUDA isn't abailable")
        exit(-1)
    main()

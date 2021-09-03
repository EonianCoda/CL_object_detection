from torch import tensor
from IL_method.persuado_label import Labeler
import argparse
import collections
from retinanet.losses import calc_iou
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from evaluator import Evaluator
# torch 
import torch
import torch.optim as optim
from torchvision import transforms
# retinanet
from retinanet.model import create_retinanet
from retinanet.dataloader import AspectRatioBasedSampler, IL_dataset, collater
from retinanet.dataloader import Resizer, Augmenter, Normalizer
# preprocessing
from preprocessing.params import Params
# train
import time
import numpy as np
# Global Setting

ROOT_DIR = "/home/deeplab307/Documents/Anaconda/Shiang/IL/" 
PRINT_INFO = True # whether print some information about continual learning on the scrren
DEFAULT_ALPHA = 0.25
DEFAULT_GAMMA = 2.0
DEFAULT_BATCH_SIZE = 5

class SimpleFocalLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations, params, cur_state:int):
        alpha = 0.25
        gamma = 2

        batch_size = classifications.shape[0]
        bg_losses = []
        fg_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights

        past_class_num = params.states[cur_state]['num_past_class']
        if params['enhance_on_new']:
            enhance_on_new_loss = torch.tensor(0).float().cuda()

        for j in range(batch_size):
            classification = classifications[j, :, :] # shape = (num_anchors, class_num)
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                alpha_factor = torch.ones(classification.shape, device=torch.device('cuda:0')) * alpha

                alpha_factor = 1. - alpha_factor
                focal_weight = classification
                focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                bce = -(torch.log(1.0 - classification))

                cls_loss = focal_weight * bce

                bg_losses.append(cls_loss.sum())
                fg_losses.append(torch.tensor(0).float().cuda())
                regression_losses.append(torch.tensor(0).float().cuda())
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # shape=(num_anchors, num_annotations)
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # shape=(num_anchors x 1)
            
            # compute the loss for classification
            targets = torch.ones(classification.shape, device=torch.device('cuda:0')) * -1
    
            # get background anchor idx
            bg_mask = torch.lt(IoU_max, 0.4)
            # whether ignore past class
            if not params['ignore_past_class']:
                targets[bg_mask, :] = 0
            else:
                targets[bg_mask, past_class_num:] = 0

            positive_indices = torch.ge(IoU_max, 0.5) 

            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            
            alpha_factor = torch.ones(targets.shape , device=torch.device('cuda:0')) * alpha

            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification) #shape = (Anchor_num, class_num)


            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            
            cls_loss = focal_weight * bce
            
            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape, device=torch.device('cuda:0')))
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            if params['enhance_on_new']:
                # false negative mask on new task
                fn_mask = classification[bg_mask, past_class_num:] > 0.05
                if fn_mask.sum() != 0:
                    enhance_on_new_loss += torch.pow(classification[bg_mask, past_class_num:][fn_mask], 2).sum() / fn_mask.sum()
            
            bg_losses.append(cls_loss[torch.eq(targets, 0.0)].sum() /torch.clamp(num_positive_anchors.float(), min=1.0))
            fg_losses.append(cls_loss[torch.eq(targets, 1.0)].sum() /torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths  = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x   = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y   = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths  = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)

                targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if torch.cuda.is_available():
                    # targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                    targets = targets/ torch.cuda.FloatTensor([[0.1, 0.1, 0.2, 0.2]])
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                regression_losses.append(torch.tensor(0).float().cuda())

        result = {'cls_loss': (torch.stack(bg_losses), torch.stack(fg_losses)),
                  'reg_loss': torch.stack(regression_losses).mean(dim=0, keepdim=True)}
        
        # if incremental_state:
        #     if params['distill']:
        #         result['bg_masks'] = torch.cat(bg_masks)
        if params['enhance_on_new']:
            result['enhance_on_new_loss'] = enhance_on_new_loss / classifications.shape[0]
        return result


def fast_zero_grad(model):
    for param in model.parameters():
        param.grad = None

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_tools(params:Params):
    start_state = params['start_state']
    start_epoch = params['start_epoch']

    # Create the model
    if start_epoch == 1 and start_state != 0:
        retinanet = create_retinanet(params['depth'], params.states[start_state - 1]['num_knowing_class'])
    else:
        retinanet = create_retinanet(params['depth'], params.states[start_state]['num_knowing_class'])
    retinanet = retinanet.cuda()
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=params['lr'])
    
    # Read checkpoint
    if start_state != 0 or start_epoch != 1:
        if start_epoch == 1:
            params.load_model(start_state - 1,-1,retinanet)
        else:
            params.load_model(start_state,start_epoch - 1,retinanet)
    
    # Training dataloader
    dataset_train = IL_dataset(params,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
                               start_state=start_state)
    if params['persuado_label']:
        labler = Labeler(retinanet, params)
        persuado_label = labler.get_persuado_label(params['start_state'])                          
        dataset_train.persuado_label = persuado_label

    sampler = AspectRatioBasedSampler(dataset_train, batch_size = params['batch_size'], drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)
    return retinanet, optimizer, dataloader_train

def get_parser(args=None):
    parser = argparse.ArgumentParser()
    # must set params
    parser.add_argument('--root_dir', help='the root dir for training', default=ROOT_DIR)
    parser.add_argument('--dataset', help='Dataset name, must contain name and years, for instance: voc2007,voc2012', default='voc2007')
    parser.add_argument('--start_epoch', type=int)
    parser.add_argument('--end_epoch', help='Number of epochs', type=int)
    parser.add_argument('--start_state', type=int)
    parser.add_argument('--end_state', type=int)
    # retinanet params
    parser.add_argument('--alpha', type=float, default=DEFAULT_ALPHA)
    parser.add_argument('--gamma', type=float, default=DEFAULT_GAMMA)
    # Other params
    parser.add_argument('--record', help='whether record training with tensorboard default=True', type=str2bool, default=True)  
    parser.add_argument('--debug', help='whether debug in Train process, default = False', type=str2bool, default=False)
    parser.add_argument('--val', help='whether do validation after training', type=str2bool, default=False)  

    ###############################
    # Incremental learning params #
    ###############################
    
    # Warm up
    parser.add_argument('--warm_stage', help='the number of warm-up stage, 0 mean not warm up, default = 0', type=int, default=0)
    parser.add_argument('--warm_epoch', help='the number of epoch for each warm-up stage, use " "(space) to split epoch for different stage', type=int, nargs='*', default=[10,10])
    parser.add_argument('--warm_layers', help='the layers which will be warmed up, must be "output", "resnet", "fpn", and split each stage by space " "', nargs='*', default=['output','resnet'])

    # IL params 
    parser.add_argument('--scenario', help='the scenario of states, must be "20", "19 1", "10 10", "15 1", "15 1 1 1 1"', nargs="+", default=[20])
    parser.add_argument('--shuffle_class', help='whether shuffle the class, default = False',type=str2bool , default=False)
    parser.add_argument('--distill', help='whether add distillation loss, default = False',type=str2bool , default=False)
    parser.add_argument('--enhance_on_new', type=str2bool, default=False)
    parser.add_argument('--ignore_past_class', help='when calculating the focal loss, whether ignore past class), default = False',type=str2bool , default=False)
    parser.add_argument('--decrease_positive', help="the upper score of the new class in incremental state, default=1.0",type=float , default=1.0) 
    parser.add_argument('--persuado_label', help="the upper score of the new class in incremental state, default=1.0",type=str2bool , default=False) 
    parser.add_argument('--just_train_new', type=str2bool , default=False) 
    # Record
    parser.add_argument('--description', help="description for this experiment", default="None")

    # learning rate
    parser.add_argument('--lr', help="learning rate, default=1e-5", type=float, default=1e-5)
    parser.add_argument('--scheduler_milestone', type=int, nargs="+", default=[40])
    parser.add_argument('--scheduler_decay',  help="learning rate decay for scheduler, default=0.1", type=float, default=0.1)

    # always default paras
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=DEFAULT_BATCH_SIZE)

    parser = vars(parser.parse_args(args))
    return parser

def cal_loss(data, model, params):
    result = dict()
    with torch.cuda.device(0):
        img_batch = data['img'].float().cuda()
        annotations = data['annot'].cuda()
        classifications, regressions, anchors = model(img_batch, 
                                                    return_feat=False, 
                                                    return_anchor=True, 
                                                    enable_act=True)

        loss = SimpleFocalLoss().forward(classifications, regressions, anchors, annotations, cur_state=params['start_state'], params=params)
        bg_loss, fg_loss = loss['cls_loss']
        reg_loss = loss['reg_loss'].mean()
        if params['enhance_on_new']:
            result['enhance_on_new_loss'] =  loss['enhance_on_new_loss']

        result['bg_cls_loss'] = bg_loss.mean()
        result['fg_cls_loss'] = fg_loss.mean()
        result['reg_loss'] = reg_loss

        return result


def main(args=None):
    parser = get_parser(args)
    params = Params(parser)

    model, optimizer, dataloader_train = create_tools(params)

    model.train()
    model.freeze_layers(['classificationModel.output','regressionModel.output'])


    loss_hist = collections.deque(maxlen=500)

    classificationModel = model.classificationModel
    cur_state = params['start_state']
    num_classes = params.states[cur_state]['num_knowing_class']
    num_old_classes = params.states[cur_state]['num_past_class']
    num_anchors = classificationModel.num_anchors

    print("Total IterNum:",len(dataloader_train))
    for epoch in range(params['start_epoch'], params['end_epoch'] + 1):
        for iter_num, data in enumerate(dataloader_train):
            start = time.time()
            fast_zero_grad(model)

            with torch.cuda.device(0):
                try:
                    result = cal_loss(data, model, params)
                    loss = torch.tensor(0).float().cuda()

                    # collect loss
                    info = [epoch, iter_num]
                    output = 'Epoch: {0[0]:2d} | Iter: {0[1]:3d}'
                    for key, value in result.items():
                        loss += value
                        output += ' | {0[%d]}: {0[%d]:1.4f}' % (len(info), len(info)+1)
                        info.extend([key, value])
                    
                    output += ' | Running loss: {0[%d]:1.5f} | Spend Time:{0[%d]:1.2f}s' % (len(info), len(info)+1)
                    end = time.time()
                    info.extend([np.mean(loss_hist), float(end - start)])

                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    # ignore grad for old class
                    if params['just_train_new']:
                        for i in range(num_anchors):
                            start_idx = i *  num_classes
                            classificationModel.output.weight.grad[start_idx : start_idx + num_old_classes,:,:,:] = 0
                            classificationModel.output.bias.grad[start_idx : start_idx + num_old_classes] = 0
                        
                    optimizer.step()
                    loss_hist.append(float(loss))
                    end = time.time()

                    print(output.format(info))
                except Exception as e:
                    print(e)
                    return None
        params.save_checkpoint(params['start_state'],epoch, model)
        if epoch % 5 == 0:
            params.auto_delete(params['start_state'],epoch)

   

if __name__ == '__main__':
    assert torch.__version__.split('.')[0] == '1'
    if not torch.cuda.is_available():
        print("CUDA isn't abailable")
        exit(-1)
    main()


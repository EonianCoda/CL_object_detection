import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class FocalLoss(nn.Module):
    #def __init__(self):

    def forward(self, classifications, regressions, anchors, annotations, distill_loss = False, pre_class_num = 0, special_alpha = 1, enhance_error=False,decrease_positive=False,each_cat_loss=False, decrease_new=False):
        
        alpha = 0.5
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        
        if distill_loss and enhance_error and pre_class_num != 0:
            enhance_loss = None
        if distill_loss:
            negative = torch.Tensor([]).long().cuda()
#         else:
#             alpha = 0.5
        for j in range(batch_size):

            classification = classifications[j, :, :] #shape = (Anchor_num, class_num)
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())
                    
                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())
                    
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1
            

            # compute the loss for classification
            targets = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                targets = targets.cuda()
            
            positive_indices = torch.ge(IoU_max, 0.5)
#             if not decrease_new:
#                 positive_indices = torch.ge(IoU_max, 0.5)
#             else:
#                 positive_indices = torch.ge(IoU_max, 0.6)
                
                
            if not distill_loss:
                targets[torch.lt(IoU_max, 0.4), :] = 0
            else:
                targets[torch.lt(IoU_max, 0.4), pre_class_num:] = 0
#                 targets[torch.lt(IoU_max, 0.4), :] = 0
#                 if not decrease_new:
#                     targets[torch.lt(IoU_max, 0.4), pre_class_num:] = 0
#                 else:
#                     targets[torch.lt(IoU_max, 0.5), pre_class_num:] = 0

                #negative = torch.cat((negative, torch.lt(IoU_max, 0.4).reshape(1,-1)))
            
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            
            
            
            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() * alpha #alpha = 0.25
            else: 
                alpha_factor = torch.ones(targets.shape) * alpha
            
            if special_alpha == 1:
                alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor) #shape = (Anchor_num, class_num)
            else:  
                print('special_alpha:',special_alpha)
                alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, (1. - alpha_factor)*special_alpha) #shape = (Anchor_num, class_num)
             
            if not decrease_positive:
                focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification) #shape = (Anchor_num, class_num)
            else:
                print('decrease_positive!')
                focal_weight = torch.where(torch.eq(targets, 1.), 0.7 - torch.clip(classification, 0, 0.7), classification)
                
                
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            
        
            
            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce
            
            ###################
            #  enhance_error  #
            ###################
            if distill_loss and enhance_error and pre_class_num != 0:
                temp = classification[torch.lt(IoU_max, 0.4), pre_class_num:]
                if (temp > 0.05).sum() != 0:
                    if enhance_loss == None:
                        enhance_loss = torch.pow(temp[temp > 0.05], 2).sum()
                    else:
                        enhance_loss += torch.pow(temp[temp > 0.05], 2).sum()
                    
#             if not distill_loss and enhance_error and pre_class_num != 0:
#                 torch.pow(classification[:,pre_class_num:], 2).sum()
#                 print('enhace_error!')
                
                
                #cls_loss[:,pre_class_num:] *= 2
                
            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            
            
            
            if not each_cat_loss:
                classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))
            else:
                cls_loss = cls_loss[positive_indices,:].mean(dim=1)
                cat_loss = defaultdict(list)
                categories = assigned_annotations[positive_indices, 4].long()
                for idx in range(categories.shape[0]):
                    cat_loss[int(categories[idx])].append(float(cls_loss[idx]))
                return cat_loss, 0

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
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets/torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                
                if not each_cat_loss:
                    regression_losses.append(regression_loss.mean())
                else:
                    regression_losses.append(regression_loss.mean(dim=0))
               
            else:
                if torch.cuda.is_available():
                    #regression_losses.append(torch.tensor(0).float().cuda())
                    if not each_cat_loss:
                        regression_losses.append(torch.tensor(0).float().cuda())
                    else:
                        
                        regression_losses.append(torch.zeros(classification.shape[-1]).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())
                    
#         if not each_cat_loss:
#             return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
#         else:
#             del regression_losses
#             return classification_losses

        
        if enhance_error and pre_class_num != 0:
            if not distill_loss: 
                classifications = classifications[:,:,pre_class_num:]
                #enhance_loss = torch.abs(classifications[classifications > 0.05]).sum() #L1 loss
                enhance_loss = torch.pow(classifications[classifications > 0.05], 2).sum() / classifications.shape[0]
            elif distill_loss:
                if enhance_loss != None:
                    enhance_loss /= classifications.shape[0]
            print('enhace_error!')
            return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True), enhance_loss 
        else:        
            return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
#         if not distill_loss:
#             return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True)
#         else:
#             return torch.stack(classification_losses).mean(dim=0, keepdim=True), torch.stack(regression_losses).mean(dim=0, keepdim=True) , negative
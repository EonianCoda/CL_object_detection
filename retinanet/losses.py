import torch
import torch.nn as nn

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
    def forward(self, classifications, regressions, anchors, annotations, cur_state:int,params):
        alpha = params['alpha'] # default = 0.25
        gamma = params['gamma'] # default = 2

        # whether the state > 0, mean it is incremental state
        if cur_state > 0:
            incremental_state = True
            if params['ignore_ground_truth']:
                non_GD = torch.Tensor([]).long().cuda()
        else:
            incremental_state = False

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        

        for j in range(batch_size):

            classification = classifications[j, :, :] #shape = (num_anchors, class_num)
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            
            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape, device=torch.device('cuda:0')) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())
                    
                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())
                    
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # shape=(num_anchors, num_annotations)
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # shape=(num_anchors x 1)
            
            # compute the loss for classification
            targets = torch.ones(classification.shape, device=torch.device('cuda:0')) * -1

            # if torch.cuda.is_available():
            #     targets = targets.cuda()
            
            # whether ignore past class
            if not incremental_state or (incremental_state and not params['ignore_past_class']):
                targets[torch.lt(IoU_max, 0.4), :] = 0
            else:
                past_class_num = params.states[cur_state]['num_past_class']
                targets[torch.lt(IoU_max, 0.4), past_class_num:] = 0

                if params['ignore_ground_truth']:
                    non_GD = torch.cat((non_GD, torch.lt(IoU_max, 0.4).reshape(1,-1)))
            
            positive_indices = torch.ge(IoU_max, 0.5) 
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1
            
            
            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape , device=torch.device('cuda:0')) * alpha
                # alpha_factor = torch.ones(targets.shape).cuda() * alpha
            else: 
                alpha_factor = torch.ones(targets.shape) * alpha
            
           
            if not incremental_state:
                focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification) #shape = (Anchor_num, class_num)
            else:
                new_class_upper_score = params['decrease_positive']
                focal_weight = torch.where(torch.eq(targets, 1.), new_class_upper_score - torch.clip(classification, 0, new_class_upper_score), classification)
                
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            
            cls_loss = focal_weight * bce
            
            ################################
            #  enhance_error for new class #
            ################################
            # if incremental_state and enhance_error and pre_class_num != 0:
            #     temp = classification[torch.lt(IoU_max, 0.4), pre_class_num:]
            #     if (temp > 0.05).sum() != 0:
            #         if enhance_loss == None:
            #             enhance_loss = torch.pow(temp[temp > 0.05], 2).sum()
            #         else:
            #             enhance_loss += torch.pow(temp[temp > 0.05], 2).sum()
                    
            # if not incremental_state and enhance_error and pre_class_num != 0:
            #     torch.pow(classification[:,pre_class_num:], 2).sum()
            #     print('enhace_error!')
            #     cls_loss[:,pre_class_num:] *= 2
                
            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape, device=torch.device('cuda:0')))
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

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

                #non_GD_indices = 1 + (~positive_indices)

                regression_diff = torch.abs(targets - regression[positive_indices, :])

                regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                
                regression_losses.append(regression_loss.mean())
            else:
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        result = [torch.stack(classification_losses).mean(dim=0, keepdim=True), 
                  torch.stack(regression_losses).mean(dim=0, keepdim=True)]

        if incremental_state and params['ignore_ground_truth']:
            result.append(non_GD)

        return tuple(result)

class IL_Loss():
    def __init__(self, il_trainer):
        
        self.model = il_trainer.model
        self.il_trainer = il_trainer
        self.params = il_trainer.params
        self.focal_loss = FocalLoss()
        self.act = nn.Sigmoid()
        self.smoothL1Loss = nn.SmoothL1Loss()
    def forward(self, img_batch, annotations, is_replay=False):
        """
            Args:
                img_batch: a tenor for input
                annotations: the annotation for img_batch
                is_replay: whether the data is from replay dataset, default=False
        """

        ##############
        # init para  #
        ##############
        
        cur_state = self.il_trainer.cur_state
        past_class_num = self.il_trainer.params.states[cur_state]['num_past_class']

        # whether calculate the distillation loss with logits 
        if self.il_trainer.params['distill']:
            distill_logits = self.il_trainer.params['distill_logits']
        else:
            distill_logits = False
        
        # whether the model is in warm up, and warm up on classifier
        cur_warm_stage = self.il_trainer.cur_warm_stage
        if cur_warm_stage != -1 and self.params['warm_layers'][cur_warm_stage] == 'output':
            classifier_warm_stage = True
        else:
            classifier_warm_stage = False

        if cur_state > 0 and not is_replay and not classifier_warm_stage:
            increment_state = True
        else:
            increment_state = False

        result = {}


        #################
        # Start forward #
        #################

        # non-incremental state
        if not increment_state:
            classification, regression, anchors = self.il_trainer.model(img_batch, 
                                                                return_feat=False, 
                                                                return_anchor=True, 
                                                                enable_act=True)
            cls_loss, reg_loss = self.focal_loss(classification, regression, anchors, annotations, 0, self.params)
            result['cls_loss'] = cls_loss.mean()
            result['reg_loss'] = reg_loss.mean()
            # Enhance error on Replay dataset
            if self.il_trainer.params['enhance_error'] and is_replay:
                classification = classification[:,:,past_class_num:]
                classification = classification[classification > 0.05]
                method = (self.il_trainer.params['enhance_error_method']).upper()
                if method == "L1":
                    enhance_loss = torch.abs(classification)
                elif method == "L2":
                    enhance_loss = torch.pow(classification, 2)
                elif method == "L3":
                    enhance_loss = torch.pow(classification, 3)
                enhance_loss = enhance_loss.sum() / max(classification.shape[0], 1)

                result['enhance_loss'] = enhance_loss
        # incremental state
        else:
            classification, regression, features, anchors = self.il_trainer.model(img_batch, 
                                                                                return_feat=True, 
                                                                                return_anchor=True, 
                                                                                enable_act=False)
            # Compute focal loss
            losses = self.focal_loss(self.act(classification), regression, anchors, annotations,cur_state,self.params)

            # Whether ignore ground truth           
            if self.params['ignore_ground_truth']:
                cls_loss, reg_loss, non_GD = losses
            else:
                cls_loss, reg_loss = losses
            result['cls_loss'] = cls_loss.mean()
            result['reg_loss'] = reg_loss.mean()
            
            # Compute distillation loss
            if self.params['distill']:
                with torch.no_grad():
                    prev_classification, prev_regression, prev_features = self.il_trainer.prev_model(img_batch,
                                                                                    return_feat=True, 
                                                                                    return_anchor=False, 
                                                                                    enable_act=False)
                
                dist_feat_loss = torch.cat([self.smoothL1Loss(prev_features[i], features[i]).view(1) for i in range(len(features))])
                dist_feat_loss = dist_feat_loss.mean()


                # Ingore the result of the new class
                classification = classification[:,:,:past_class_num]

                # Ignore ground truth
                if self.params['ignore_ground_truth']:
                    non_GD = torch.flatten(non_GD)

                    prev_classification = prev_classification.view(-1, past_class_num)[non_GD,:]
                    classification = classification.view(-1, past_class_num)[non_GD, :]
                    prev_regression = prev_regression.view(-1,4)[non_GD,:]
                    regression = regression.view(-1,4)[non_GD,:]

                    if distill_logits:
                        greater = self.act(prev_classification) > 0.05
                    else:
                        prev_classification = self.act(prev_classification)
                        classification = self.act(classification)
                        greater = prev_classification > 0.05

                    dist_reg_loss = self.smoothL1Loss(prev_regression[greater.any(dim=2)], regression[greater.any(dim=2)])
                    dist_class_loss = nn.MSELoss()(prev_classification[greater], classification[greater])

                else:
                    # whether compute distillation loss with logits
                    if distill_logits:
                        if self.il_trainer.params['distill_logits_on'] == "new":
                            greater = self.act(classification) > 0.05
                        else:
                            greater = self.act(prev_classification) > 0.05
                    else:
                        prev_classification = self.act(prev_classification)
                        classification = self.act(classification)
                        greater = prev_classification > 0.05
                

                    dist_reg_loss = self.smoothL1Loss(prev_regression[greater.any(dim=2)], regression[greater.any(dim=2)])
                    dist_class_loss = nn.MSELoss()(prev_classification[greater], classification[greater])

                    # add background loss if distill with logits
                    if distill_logits and self.il_trainer.params['distill_logits_bg_loss']:
                        bg_class_loss = nn.MSELoss()(prev_classification[~greater], classification[~greater])
                        # bg_class_loss /= torch.numel(prev_classification)
                        dist_class_loss += bg_class_loss
                    
                result['dist_cls_loss'] = dist_class_loss
                result['dist_reg_loss'] = dist_reg_loss
                result['dist_feat_loss'] = dist_feat_loss
    
            # compute MAS loss
            if self.params['mas']:
                mas_loss = self.il_trainer.mas.penalty(self.il_trainer.prev_model)
                result['mas_loss'] = mas_loss



        return result
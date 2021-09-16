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

class ProtoTypeFocalLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations, cur_state:int,params, cls_features, prototype_features):
        def _distance(a, b):
            return torch.norm(a - b, dim=2)
        alpha = params['alpha'] # default = 0.25
        gamma = params['gamma'] # default = 2
        num_anchors = 9

        # whether the state > 0, mean it is incremental state
        incremental_state = (cur_state > 0)
        if incremental_state:
            # if params['enhance_on_new']:
            #     enhance_loss_on_new = None
            if params['distill']:
                bg_masks = []

        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        
        pos_indices = []
        pos_targets = []
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
                classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0).float().cuda())
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # shape=(num_anchors, num_annotations)
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # shape=(num_anchors x 1)
            

            pos_indices.append(torch.ge(IoU_max, 0.5).view(-1, num_anchors).unsqueeze(dim=0))
            target = bbox_annotation[IoU_argmax, 4].view(-1, num_anchors)
            pos_targets.append(target.long().unsqueeze(dim=0))


            # compute the loss for classification
            targets = torch.ones(classification.shape, device=torch.device('cuda:0')) * -1
    
            # get background anchor idx
            bg_mask = torch.lt(IoU_max, 0.4)
            # whether ignore past class
            if not incremental_state or (incremental_state and not params['ignore_past_class']):
                targets[bg_mask, :] = 0
            else:
                past_class_num = params.states[cur_state]['num_past_class']
                targets[bg_mask, past_class_num:] = 0
                if params['new_ignore_past_class']:
                    # targets_old = torch.zeros(classification.shape, device=torch.device('cuda:0'))
                    # targets_old[bg_mask, :past_class_num] = 1

                    old_prod = torch.sum(classification[:, :past_class_num], dim=1)
                    targets[torch.logical_and(bg_mask, old_prod < 0.5), :past_class_num] = 0
                    #torch.log(old_prod) * (1 - torch.clamp(old_prod, max=1)) 

            positive_indices = torch.ge(IoU_max, 0.5) 

            # store non positive anchors for distillation loss
            if incremental_state and params['distill']:
                bg_masks.append((~positive_indices).unsqueeze(dim=0))

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
            elif params['decrease_positive_by_IOU']:
                mid_indices = torch.logical_and(torch.le(IoU_max, 0.7), positive_indices)

                focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)

                targets_for_mid = torch.zeros(classification.shape, device=torch.device('cuda:0'))
                targets_for_mid[mid_indices, assigned_annotations[mid_indices, 4].long()] = 1
                
                upper_score = torch.clip(IoU_max + 0.2, 1e-4, 1 - 1e-4).unsqueeze(dim=1)
                focal_weight = torch.where(torch.eq(targets_for_mid, 1), torch.where(classification >= upper_score, torch.ones(classification.shape, device=torch.device('cuda:0')) * 1e-4, torch.abs(classification - upper_score)), focal_weight)

            else:
                new_class_upper_score = params['decrease_positive']
                focal_weight = torch.where(torch.eq(targets, 1.), new_class_upper_score - torch.clip(classification, 0, new_class_upper_score), classification)


            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            
            cls_loss = focal_weight * bce
            
            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape, device=torch.device('cuda:0')))
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            bg_losses.append(cls_loss[torch.eq(targets, 0.0)].sum() /torch.clamp(num_positive_anchors.float(), min=1.0))
            fg_losses.append(cls_loss[torch.eq(targets, 1.0)].sum() /torch.clamp(num_positive_anchors.float(), min=1.0))

            # classification_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))


            # if incremental_state and params['enhance_on_new']:
            #     new_class_bg = classification[bg_mask, past_class_num:]
            #     # false negative mask
            #     fn_mask = new_class_bg > 0.05
            #     if (fn_mask > 0.05).sum() != 0:
            #         if enhance_loss_on_new == None:
            #             enhance_loss_on_new = torch.pow(new_class_bg[fn_mask], 2).sum() /torch.clamp(num_positive_anchors.float(), min=1.0)
            #         else:
            #             enhance_loss_on_new += torch.pow(new_class_bg[fn_mask], 2).sum() /torch.clamp(num_positive_anchors.float(), min=1.0)

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
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        # cal prototype_loss
        pos_indices = torch.cat(pos_indices) #shape = (batch_size, all_anchor_num / 9, 9)
        pos_targets = torch.cat(pos_targets)
        
        mask = pos_indices.any(dim=2)
        pos_indices = pos_indices[mask,:]
        pos_targets = pos_targets[mask,:]

        cls_features = cls_features[mask] # shape = (num_pos_anchor, channels)   
        
        num_new_classes = len(params.states[cur_state]['new_class']['id'])
        count = torch.zeros(num_new_classes, num_anchors, 1).cuda()
        cur_prototype_features = torch.zeros(num_new_classes, num_anchors, cls_features.shape[1]).cuda()
        pos_targets -= past_class_num
        
        # for class_id in range(past_class_num):
        for i in range(cls_features.shape[0]):
            count[pos_targets[i][pos_indices[i]], pos_indices[i],:] += 1
            cur_prototype_features[pos_targets[i][pos_indices[i]], pos_indices[i],:] += cls_features[i]

        cur_prototype_features /= torch.clamp(count,min=1)
        cur_prototype_features = torch.mean(cur_prototype_features, dim=1).unsqueeze(dim=1)
        distance = _distance(cur_prototype_features.view(-1, cls_features.shape[1]), prototype_features)
        prototype_loss = torch.clamp(600 - distance, min=0).mean() * 0.1

        # if (prototype_loss == 0).sum() == 0:
        #     prototype_loss = torch.tensor(0).float().cuda()
        # else:
        #     prototype_loss = torch.mean(prototype_loss[prototype_loss != 0])

        result = {'cls_loss': torch.stack(classification_losses),
                  'reg_loss': torch.stack(regression_losses).mean(dim=0, keepdim=True),
                  'prototype_loss': prototype_loss}
        
        if incremental_state:
            if params['distill']:
                result['bg_masks'] = torch.cat(bg_masks)
        return result

class FocalLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations, cur_state:int,params, progress=-1):
        alpha = params['alpha'] # default = 0.25
        gamma = params['gamma'] # default = 2

        # whether the state > 0, mean it is incremental state
        incremental_state = (cur_state > 0)
            
       
        if incremental_state:
            # if params['enhance_on_new']:
            #     enhance_loss_on_new = None
            if params['distill']:
                bg_masks = []
            if params['enhance_on_new']:
                enhance_on_new_loss = torch.tensor(0).float().cuda()
 

        batch_size = classifications.shape[0]
        bg_losses = []
        fg_losses = []
        # classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]
        anchor_widths  = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x   = anchor[:, 0] + 0.5 * anchor_widths
        anchor_ctr_y   = anchor[:, 1] + 0.5 * anchor_heights
        

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
                # classification_losses.append(cls_loss.sum())
                regression_losses.append(torch.tensor(0).float().cuda())
                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # shape=(num_anchors, num_annotations)
            IoU_max, IoU_argmax = torch.max(IoU, dim=1) # shape=(num_anchors x 1)
            
            # compute the loss for classification
            targets = torch.ones(classification.shape, device=torch.device('cuda:0')) * -1
    
            # get background anchor idx
            bg_mask = torch.lt(IoU_max, 0.4)
            past_class_num = params.states[cur_state]['num_past_class']
            # whether ignore past class
            if not incremental_state or (incremental_state and not params['ignore_past_class']):
                targets[bg_mask, :] = 0
            else:
                targets[bg_mask, past_class_num:] = 0
                if params['new_ignore_past_class']:
                    # targets_old = torch.zeros(classification.shape, device=torch.device('cuda:0'))
                    # targets_old[bg_mask, :past_class_num] = 1

                    old_prod = torch.sum(classification[:, :past_class_num], dim=1)
                    targets[torch.logical_and(bg_mask, old_prod < 0.5), :past_class_num] = 0
                    #torch.log(old_prod) * (1 - torch.clamp(old_prod, max=1)) 

            positive_indices = torch.ge(IoU_max, 0.5) 

            # store non positive anchors for distillation loss
            if incremental_state and params['distill']:
                bg_masks.append((~positive_indices).unsqueeze(dim=0))

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
            elif params['decrease_positive_by_IOU']:
                mid_indices = torch.logical_and(torch.le(IoU_max, 0.7), positive_indices)

                focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification, classification)

                targets_for_mid = torch.zeros(classification.shape, device=torch.device('cuda:0'))
                targets_for_mid[mid_indices, assigned_annotations[mid_indices, 4].long()] = 1
                
                upper_score = torch.clip(IoU_max + 0.2, 1e-4, 1 - 1e-4).unsqueeze(dim=1)
                focal_weight = torch.where(torch.eq(targets_for_mid, 1), torch.where(classification >= upper_score, torch.ones(classification.shape, device=torch.device('cuda:0')) * 1e-4, torch.abs(classification - upper_score)), focal_weight)

            else:
                new_class_upper_score = params['decrease_positive']
                focal_weight = torch.where(torch.eq(targets, 1.), new_class_upper_score - torch.clip(classification, 0, new_class_upper_score), classification)


            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)
            bce = -(targets * torch.log(classification) + (1.0 - targets) * torch.log(1.0 - classification))
            
            cls_loss = focal_weight * bce
            
            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape, device=torch.device('cuda:0')))
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            
            if incremental_state and params['enhance_on_new']:
                # false negative mask on new task
                fn_mask = classification[bg_mask, past_class_num:] > 0.05
                if fn_mask.sum() != 0:
                    enhance_on_new_loss += torch.pow(classification[bg_mask, past_class_num:][fn_mask], 2).sum()
                    # print(cls_loss[bg_mask, past_class_num:][fn_mask].sum())
            
            # fake label
            if incremental_state and params['persuado_label'] and progress != -1:
                fake_label_anchor = (targets[:,past_class_num:] == 1).any(dim=1)
                # false positive for old class in new target
                fp_mask = classification[fake_label_anchor, :past_class_num] > 0.05
                cls_loss[fake_label_anchor, :past_class_num][fp_mask] *= progress


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
                if torch.cuda.is_available():
                    regression_losses.append(torch.tensor(0).float().cuda())
                else:
                    regression_losses.append(torch.tensor(0).float())

        result = {'cls_loss': (torch.stack(bg_losses), torch.stack(fg_losses)),
                  'reg_loss': torch.stack(regression_losses).mean(dim=0, keepdim=True)}
        
        if incremental_state:
            if params['distill']:
                result['bg_masks'] = torch.cat(bg_masks)
            if params['enhance_on_new']:
                result['enhance_on_new_loss'] = enhance_on_new_loss
        return result

class IL_Loss():
    def __init__(self, il_trainer):
        
        self.model = il_trainer.model
        self.il_trainer = il_trainer
        self.params = il_trainer.params
        self.focal_loss = FocalLoss()
        self.classifier_act = nn.Sigmoid()
        self.smoothL1Loss = nn.SmoothL1Loss()


        if self.params['prototype_loss']:
            self.prototypefocal_loss = ProtoTypeFocalLoss()
            # if self.il_trainer.protoTyper.prototype_features == None:
            #     self.il_trainer.protoTyper.init_prototype(self.il_trainer.cur_state - 1)
            _ , _ , feature_channels = self.il_trainer.protoTyper.prototype_features.shape    
            self.il_trainer.protoTyper.prototype_features = torch.mean(self.il_trainer.protoTyper.prototype_features,dim=1).unsqueeze(dim=0).cuda()

        # calculate past classifer'norm 
        if self.params['classifier_loss']:
            classifier = self.il_trainer.prev_model.classificationModel.output.weight.data
            num_anchors = self.il_trainer.prev_model.classificationModel.num_anchors
            num_prev_classes = self.il_trainer.prev_model.num_classes

            self.past_classifer = []
            self.past_classifier_norm = []
            for class_idx in range(num_prev_classes):
                indices = [i * num_prev_classes + class_idx for i in range(num_anchors)]

                indices = torch.tensor(indices).long().cuda()
                self.past_classifer.append(torch.index_select(classifier, 0, indices).flatten().unsqueeze(dim=0))
                self.past_classifier_norm.append(torch.norm(self.past_classifer[-1]).unsqueeze(dim=0))

            
            self.past_classifer = torch.cat(self.past_classifer).cuda()
            self.past_classifier_norm = torch.cat(self.past_classifier_norm).cuda()

    def cal_classifier_loss(self, delta=0.5):

        if self.params['classifier_loss'] == False:
            raise ValueError("Please enable classifier loss first")

        num_anchors = self.il_trainer.prev_model.classificationModel.num_anchors
        num_classes = self.il_trainer.model.num_classes
        num_prev_classes = self.il_trainer.prev_model.num_classes
        num_new_classes = num_classes - num_prev_classes

        cur_classifier = self.il_trainer.model.classificationModel.output.weight.data

        sim_loss = torch.tensor(0).float().cuda()
        for class_idx in range(num_new_classes):
            indices = [i * num_classes + num_prev_classes + class_idx for i in range(num_anchors)]
            indices = torch.tensor(indices).long().cuda()
            w = torch.index_select(cur_classifier, 0, indices).flatten()
            
            loss = torch.mul(w, self.past_classifer).sum(dim=1) / (self.past_classifier_norm * torch.norm(w))
            loss = torch.sum(torch.clamp(loss.abs() - delta, min=0))
            sim_loss += loss
            
        return sim_loss
     
    def forward(self, img_batch, annotations, is_replay=False, is_bic=False):
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
            # Bic method
            if self.params['bic']:
                classification, regression, anchors = self.il_trainer.model(img_batch, 
                                                                    return_feat=False, 
                                                                    return_anchor=True, 
                                                                    enable_act=False)
                classification = self.il_trainer.bic.bic_correction(classification)
                classification = self.classifier_act(classification)
            else:
                classification, regression, anchors = self.il_trainer.model(img_batch, 
                                                                    return_feat=False, 
                                                                    return_anchor=True, 
                                                                    enable_act=True)
            losses = self.focal_loss(classification, regression, anchors, annotations, 0, self.params)

            # clip too small loss
            if self.il_trainer.params['clip_loss'] and is_replay:
                result['cls_bg_loss'], result['cls_fg_loss']  = losses['cls_loss']
                mask = result['cls_fg_loss'] >= self.il_trainer.params['clip_replay_cls_loss']
                if mask.sum() == 0:
                    result['cls_fg_loss'] = torch.tensor(0).float().cuda()
                else:
                    result['cls_fg_loss'] = result['cls_fg_loss'][mask].mean()
                result['cls_bg_loss'] = result['cls_bg_loss'].mean()
            else:
                result['cls_bg_loss'], result['cls_fg_loss']  = losses['cls_loss']
                result['cls_bg_loss'] = result['cls_bg_loss'].mean()
                result['cls_fg_loss'] = result['cls_fg_loss'].mean()

            result['reg_loss'] = losses['reg_loss'].mean()

            # Enhance error on Replay dataset
            if self.il_trainer.params['enhance_error'] and is_replay and is_bic == False:
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
            if self.il_trainer.params['prototype_loss'] and self.il_trainer.cur_epoch > 5:
                classification, regression, features, anchors, cls_features = self.il_trainer.model.forward_prototype(img_batch, 
                                                                                                        return_feat=True, 
                                                                                                        return_anchor=True, 
                                                                                                        enable_act=False)
            else:
                classification, regression, features, anchors = self.il_trainer.model(img_batch, 
                                                                                    return_feat=True, 
                                                                                    return_anchor=True, 
                                                                                    enable_act=False)
                                                                                    
            # Bic method
            if self.params['bic']:
                classification = self.il_trainer.bic.bic_correction(classification)
            
            # Compute focal loss
            if self.il_trainer.params['prototype_loss'] and self.il_trainer.cur_epoch > 5:
                losses = self.prototypefocal_loss(self.classifier_act(classification), 
                                                    regression, anchors, 
                                                    annotations,
                                                    cur_state,
                                                    self.params, 
                                                    cls_features, 
                                                    self.il_trainer.protoTyper.prototype_features)
                result['prototype_loss'] = losses['prototype_loss']
            else:
                if not self.il_trainer.params['persuado_label']:
                    losses = self.focal_loss(self.classifier_act(classification), 
                                            regression, 
                                            anchors, 
                                            annotations,
                                            cur_state,
                                            self.params)
                else:
                    finish_progress =  float(self.il_trainer.cur_epoch / self.il_trainer.end_epoch)
                    losses = self.focal_loss(self.classifier_act(classification), 
                                            regression, 
                                            anchors, 
                                            annotations,
                                            cur_state,
                                            self.params,
                                            finish_progress)

            # clip too small loss
            if self.il_trainer.params['clip_loss']:
                result['cls_bg_loss'], result['cls_fg_loss']  = losses['cls_loss']
                mask = result['cls_fg_loss'] >= self.il_trainer.params['clip_cls_loss']
                if mask.sum() == 0:
                    result['cls_fg_loss'] = torch.tensor(0).float().cuda()
                else:
                    result['cls_fg_loss'] = result['cls_fg_loss'][mask].mean()
                result['cls_bg_loss'] = result['cls_bg_loss'].mean()
            else:
                result['cls_bg_loss'], result['cls_fg_loss']  = losses['cls_loss']
                result['cls_bg_loss'] = result['cls_bg_loss'].mean()
                result['cls_fg_loss'] = result['cls_fg_loss'].mean()
            result['reg_loss'] = losses['reg_loss'].mean()

            if self.params['enhance_on_new']:
                result['enhance_on_new_loss'] = losses['enhance_on_new_loss']
            # # Whether ignore ground truth
            # if self.params['enhance_on_new']:
            #     result['enhance_loss_on_new'] = losses['enhance_loss_on_new']

            # Compute distillation loss
            if self.params['distill']:
                bg_masks = losses['bg_masks']

                if self.params['classifier_loss']:
                    # divide by batch_size
                    result['sim_loss'] = self.cal_classifier_loss()
                with torch.no_grad():
                    prev_classification, prev_regression, prev_features = self.il_trainer.prev_model(img_batch,
                                                                                                    return_feat=True, 
                                                                                                    return_anchor=False, 
                                                                                                    enable_act=False)
                

                # use cosine similarity to calculate distillation feature loss
                dist_feat_loss = None
                feat_loss_fun = nn.CosineEmbeddingLoss()
                for i in range(len(features)):
                    b, c, w, h  = features[i].shape # (batch_size, channels, weight, height)
                    num_target = b * w * h
                    if dist_feat_loss == None:
                        dist_feat_loss = feat_loss_fun(prev_features[i].permute(0, 2, 3, 1).contiguous().view(-1, c),
                                                                features[i].permute(0, 2, 3, 1).contiguous().view(-1, c),
                                                                torch.ones(num_target, device=torch.device('cuda:0')))
                    else:
                        dist_feat_loss += feat_loss_fun(prev_features[i].permute(0, 2, 3, 1).contiguous().view(-1, c),
                                                                features[i].permute(0, 2, 3, 1).contiguous().view(-1, c),
                                                                torch.ones(num_target, device=torch.device('cuda:0')))
                # use smoothL1loss to calculate distillation feature loss

                # dist_feat_loss = torch.cat([self.smoothL1Loss(prev_features[i], features[i]).view(1) for i in range(len(features))])
                # dist_feat_loss = dist_feat_loss.mean()


                # Ingore the result of the new class
                classification = classification[:,:,:past_class_num]


                
                # whether compute distillation loss with logits
                if distill_logits:
                    # foreground which is predicted from prev model 
                    prev_fg_mask = self.classifier_act(prev_classification) > 0.05
                else:
                    prev_classification = self.classifier_act(prev_classification)
                    classification = self.classifier_act(classification)
                    # foreground which is predicted from prev model 
                    prev_fg_mask = prev_classification > 0.05


                reg_mask = torch.logical_and(bg_masks, prev_fg_mask.any(dim=2))
                dist_reg_loss = self.smoothL1Loss(prev_regression[reg_mask], regression[reg_mask])
                # dist_reg_loss = self.smoothL1Loss(prev_regression[greater.any(dim=2)], regression[greater.any(dim=2)])

                if self.params['ignore_GD']:
                    dist_class_loss = nn.MSELoss()(prev_classification[reg_mask], classification[reg_mask])
                else:
                    dist_class_loss = nn.MSELoss()(prev_classification[prev_fg_mask], classification[prev_fg_mask])

                # add background loss if distill with logits
                # if distill_logits and self.il_trainer.params['distill_logits_bg_loss']:
                #     bg_class_loss = nn.MSELoss()(prev_classification[~greater], classification[~greater])
                #     # bg_class_loss /= torch.numel(prev_classification)
                #     dist_class_loss += bg_class_loss
                    
                result['dist_cls_loss'] = dist_class_loss
                result['dist_reg_loss'] = dist_reg_loss
                result['dist_feat_loss'] = dist_feat_loss
    
        return result
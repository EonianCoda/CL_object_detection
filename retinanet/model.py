import torch.nn as nn
import torch
import math
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
from retinanet import losses
import copy
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_features_in = num_features_in
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.feature_size = feature_size
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1) #num_anchors(A) * num_classes(K)
        self.output_act = nn.Sigmoid()
        self.enable_act = True
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        if self.enable_act:
            out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

    def increase_class(self, num_newClasses):
        old_classes = self.num_classes
        old_filter_num = old_classes * self.num_anchors
        
        self.num_classes += num_newClasses
        old_output = self.output
        self.output = nn.Conv2d(self.feature_size, self.num_anchors * self.num_classes, kernel_size=3, padding=1)

        prior = 0.01 #same as variable prior at ResNet __init__()
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        
            
        for i in range(self.num_anchors):
            self.output.weight.data[i * self.num_classes:i * self.num_classes + old_classes,:,:,:] = old_output.weight.data[i * old_classes:(i+1) * old_classes,:,:,:] 
            self.output.bias.data[i * self.num_classes:i * self.num_classes + old_classes] = old_output.bias.data[i * old_classes:(i+1) * old_classes]
            
        self.output.cuda()
        del old_output
    def special_increase(self, data_loader):
        print("special_increase")
        self.num_classes = 20
        old_output = self.output
        self.output = nn.Conv2d(self.feature_size, self.num_anchors * self.num_classes, kernel_size=3, padding=1)

        
        transform = []
        for i in range(20):
            real_ID = data_loader.coco_labels[i]
            for j in range(4):
                if real_ID in data_loader.supercategory[j]:
                    transform.append(j)
                    break
                    
        for i in range(9):
            for j in range(20):
                part_idx = transform[j]  
                self.output.weight.data[i*20 + j,:,:,:] = old_output.weight.data[i*4 + part_idx,:,:,:]
                self.output.bias.data[i*20 + j] = old_output.bias.data[i*4 + part_idx]
    
        self.output.cuda()
        del old_output
class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, prev_model=None,distill_feature=False, distill_loss=False):
 
        self.num_classes = num_classes
        self.prev_num_classes = -1
        
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1] - 1].conv2.out_channels, self.layer3[layers[2] - 1].conv2.out_channels,
                         self.layer4[layers[3] - 1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]
        else:
            raise ValueError(f"Block type {block} not understood")

        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=num_classes)

        self.anchors = Anchors()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = losses.FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)

        self.freeze_bn()
        
        self.distill_feature = distill_feature
        self.distill_loss = distill_loss
        self.init_prev_model(prev_model)
        self.special_alpha = 1.0
        self.enhance_error = False
        self.decrease_positive = False
        self.each_cat_loss = False
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
    def freeze_resnet_n_fpn(self, status = False):
        if status == False:
            print('freeze resnet and fpn')
        else:
            print('unfreeze resnet and fpn')
        for name, parameter in self.named_parameters():
            if "prev_model" not in name and "regressionModel" not in name and "classificationModel" not in name:
                parameter.requires_grad = status
    def freeze_resnet(self, status = False):
        if status == False:
            print('freeze resnet')
        else:
            print('unfreeze resnet')
        for name, parameter in self.named_parameters():
            if "prev_model" not in name and "regressionModel" not in name and "classificationModel" not in name and "fpn" not in name:
                parameter.requires_grad = status
    def freeze_regression(self, status = False):
        if status == False:
            print('freeze regression')
        else:
            print('unfreeze regression')
        for name, parameter in self.named_parameters():
            if "prev_model" not in name and "classificationModel" not in name and "fpn" not in name:
                parameter.requires_grad = status
    
    def freeze_except_new_classification(self, status= False):
        """set all layers except classificationModel.output requires_grad = statsu
        """
        if status == False:
            print('freeze all layer except newclassification')
            
        else:
            print('unfreeze all layers')
        for name, parameter in self.named_parameters():
#             if "prev_model" not in name and "classificationModel.output" not in name and "regressionModel.output" not in name:
#                 parameter.requires_grad = status
            
            if "prev_model" not in name and "classificationModel.output" not in name:
                parameter.requires_grad = status
                
    def set_special_alpha(self,alpha):
        self.special_alpha = alpha
    
    
    
    def forward(self, inputs):

        if self.training and (not self.distill_feature):  
            img_batch, annotations = inputs
        else:
            img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)

        ################################
        #          Use Logits          #           
        ################################
        self.classificationModel.enable_act = (not self.distill_loss)
        if self.distill_feature:
            self.classificationModel.enable_act = False

        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1) #shape = (batch_size, W*H*A(Anchor_num), class_num)
        
        anchors = self.anchors(img_batch)

            
        if self.distill_feature:
            return (features, regression, classification)
        elif self.training:
            
            #no distill loss
            if not self.distill_loss:
                if self.prev_model:
                    if not self.enhance_error:
                        class_loss, reg_loss = self.focalLoss(classification, regression, anchors, annotations, special_alpha=self.special_alpha, enhance_error=self.enhance_error, pre_class_num=self.prev_model.num_classes, each_cat_loss = self.each_cat_loss)
                    else:
                        class_loss, reg_loss, enhance_loss = self.focalLoss(classification, regression, anchors, annotations, special_alpha=self.special_alpha, enhance_error=self.enhance_error, pre_class_num=self.prev_model.num_classes, each_cat_loss = self.each_cat_loss)
                else:
                    class_loss, reg_loss = self.focalLoss(classification, regression, anchors, annotations, special_alpha=self.special_alpha, enhance_error=self.enhance_error, each_cat_loss = self.each_cat_loss)
                    
                if not self.enhance_error:
                    return (class_loss, reg_loss)
                else:
                    return (class_loss, reg_loss, enhance_loss)
            #use distill loss
            else: 
                assert self.prev_model != None
                self.prev_model.eval()
                self.prev_model.training = False
                
                #use label
                if self.classificationModel.enable_act:
                        losses = self.focalLoss(classification, regression, anchors, annotations, self.distill_loss, self.prev_model.num_classes, special_alpha=self.special_alpha, decrease_positive=self.decrease_positive, decrease_new=True, enhance_error=self.enhance_error)
                        
                #use logits
                else:
                    classification_label = nn.Sigmoid()(classification)
                    losses = self.focalLoss(classification_label, regression, anchors, annotations, self.distill_loss, self.prev_model.num_classes, special_alpha=self.special_alpha, decrease_positive=self.decrease_positive, decrease_new=True, enhance_error=self.enhance_error)
                    
                if self.enhance_error:
                    class_loss, reg_loss, enhance_loss = losses
                else:
                    class_loss, reg_loss = losses
                ################################
                #          Ignore GD           #           
                ################################   
#                     class_loss, reg_loss, negative = self.focalLoss(nn.Sigmoid()(classification), regression, anchors, annotations, self.distill_loss, self.prev_model.num_classes,special_alpha=self.special_alpha,decrease_positive=self.decrease_positive,decrease_new=True)
                with torch.no_grad():
                    prev_features, prev_regression, prev_classification = self.prev_model(img_batch)
                
                #start disill loss
                smoothL1Loss = nn.SmoothL1Loss()
                dist_feat_loss = torch.cat([smoothL1Loss(prev_features[i], features[i]).view(1) for i in range(len(features))])
                dist_feat_loss = dist_feat_loss.mean()
                ################################
                #          Use Logits          #           
                ################################
                
                #greater = (classification_label[:,:,:self.prev_model.num_classes] > 0.05)
                #greater = (nn.Sigmoid()(prev_classification) > 0.05)
                old_label = nn.Sigmoid()(prev_classification)
                greater = (old_label > 0.05)
                #greater_class_loss = nn.MSELoss()(prev_classification[greater], classification[:,:,:self.prev_model.num_classes][greater])
#                 bg_class_loss = nn.MSELoss(reduction='sum')(prev_classification[~greater], classification[:,:,:self.prev_model.num_classes][~greater])
#                 bg_class_loss /= torch.numel(prev_classification)
                #dist_class_loss = nn.MSELoss(reduction='sum')(prev_classification, classification[:,:,:self.prev_model.num_classes])
#                 dist_class_loss = greater_class_loss + bg_class_loss
    
                dist_class_loss = (torch.pow(prev_classification - classification[:,:,:self.prev_model.num_classes], 2) * old_label).sum() / (old_label > 0.5).sum()
                #dist_class_loss /= greater.sum
                #dist_class_loss = nn.MSELoss()(prev_classification[greater], classification[:,:,:self.prev_model.num_classes][greater])
                dist_reg_loss = smoothL1Loss(prev_regression[greater.any(dim=2)], regression[greater.any(dim=2)])
                
    
                #dist_class_loss = smoothL1Loss(prev_classification, classification[:,:,:self.prev_model.num_classes])
        
        
#                 dist_class_loss = nn.MSELoss()(prev_classification, classification[:,:,:self.prev_model.num_classes])
#                 dist_reg_loss = smoothL1Loss(prev_regression, regression)

                
                ################################
                #          Ignore GD           #           
                ################################
#                 negative = torch.flatten(negative)

# #                 dist_class_loss = nn.MSELoss(reduction='sum')(prev_classification.view(-1, self.prev_model.num_classes)[negative,:], classification[:,:,:self.prev_model.num_classes].view(-1, self.prev_model.num_classes)[negative, :])
                
#                 dist_class_loss = nn.MSELoss()(prev_classification.view(-1, self.prev_model.num_classes)[negative,:], classification.view(-1, self.num_classes)[negative, :self.prev_model.num_classes])

#                 #dist_class_loss /= len(prev_classification.shape[s])
#                 dist_reg_loss = smoothL1Loss(prev_regression.view(-1,4)[negative,:], regression.view(-1,4)[negative,:])


                ################################
                #        Origin Baseline       #           
                ################################
#                 greater = torch.ge(prev_classification, 0.05)
#                 if greater.sum() != 0:
#                     dist_class_loss = nn.MSELoss()(prev_classification[greater], classification[:,:,:self.prev_model.num_classes][greater])
#                     dist_reg_loss = smoothL1Loss(prev_regression[greater.any(dim = 2)], regression[greater.any(dim = 2)])
#                 else:
#                     dist_class_loss = torch.tensor(0).float().cuda()
#                     dist_reg_loss = torch.tensor(0).float().cuda()
                
                
                ################################
                #         change ratio         #           
                ################################
#                 ratio = 1
#                 class_loss *= ratio
#                 reg_loss *= ratio
#                 dist_class_loss *= ratio
#                 dist_reg_loss *= ratio
#                 dist_feat_loss *= ratio

                if self.enhance_error:
                    return (class_loss, reg_loss, dist_class_loss, dist_reg_loss, dist_feat_loss, enhance_loss)
                else:
                    return (class_loss, reg_loss, dist_class_loss, dist_reg_loss, dist_feat_loss)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)
            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            ################################
            #  Only use the highest score  #           
            ################################            
#             scores = torch.squeeze(classification[0, :, :])
#             temp = torch.max(scores, dim=1)
#             scores = temp[0]
#             max_idxs = temp[1]
            
# #             print("scores:",scores.shape)

#             scores_over_thresh = (scores > 0.05)
            
#             anchorBoxes = torch.squeeze(transformed_anchors)
#             anchorBoxes = anchorBoxes[scores_over_thresh]
            
#             scores = scores[scores_over_thresh]
# #             print("scores_over_thresh:",scores_over_thresh.shape)
# #             print("anchorBoxes:",anchorBoxes.shape)
# #             print("scores:",scores.shape)
#             anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
            
            
#             finalResult[0].extend(scores[anchors_nms_idx])
#             finalResult[1].extend(max_idxs[anchors_nms_idx])
#             finalResult[2].extend(anchorBoxes[anchors_nms_idx])
            
#             finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
#             finalAnchorBoxesIndexesValue = max_idxs[anchors_nms_idx]#torch.tensor([i] * anchors_nms_idx.shape[0])
#             if torch.cuda.is_available():
#                 finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()
#                 finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
#                 finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
            
            ################################
            #         Origin model         #           
            ################################ 
            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                
                scores_over_thresh = (scores > 0.05)
#                 if i == classification.shape[2] - 1:
#                     scores_over_thresh = (scores > 0.2)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue
                    
                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                
                
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
                
                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()
    
                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                #print("finalAnchorBoxesIndexes",finalAnchorBoxesIndexes)
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
    
    def increase_class(self, num_newClasses, w_distillation=False):
        """add new task
        Args:
            num_newClasses: the number of new classes number
        """
        if w_distillation:
            print('Store previous model!')
            if self.prev_model != None:
                del self.prev_model
           
            prev_model = copy.deepcopy(self)
            prev_model.cuda()
            self.init_prev_model(prev_model)
        self.prev_num_classes = self.num_classes
        self.num_classes += num_newClasses
        self.classificationModel.increase_class(num_newClasses)
    
    def special_increase(self,data_loader):
        self.prev_num_classes = self.num_classes
        self.num_classes = 20
        self.classificationModel.special_increase(data_loader)
        
        
    def init_prev_model(self, prev_model):
        """init and set prev_model, then set self's distill_loss = True
        """
        
        self.prev_model = prev_model
        if self.prev_model != None:
            self.prev_model.cuda()
            self.prev_model.training = False
            self.prev_model.distill_feature = True
            self.prev_model.distill_loss = False
            self.prev_model.eval()
            
#             for p in self.parameters():
#                 p.requires_grad = False
            
            self.prev_num_classes = self.prev_model.num_classes #set previous class_num
            
            self.distill_feature = False
            self.distill_loss = True
        else:
            self.distill_feature = False
            self.distill_loss = False
        



def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], model_dir='.'), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='.'), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], model_dir='.'), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'], model_dir='.'), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'], model_dir='.'), strict=False)
    return model

import math

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
import torchvision
from retinanet import losses
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors

from preprocessing.debug import debug_print, DEBUG_FLAG
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
        
    def forward(self, x, enable_act=True):
        """
            Args:
                enable_act: whether let output layer pass activation layer, default = "true"
        """
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        if enable_act:
            out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

    def next_state(self, num_new_classes, similaritys, method="ratio"):
        """increase the number of neurons in output layer

            Args:
                num_new_classes: the number of new classes which will be added
        """
        num_old_class = self.num_classes
        #old_filter_num = num_old_class * self.num_anchors
        
        self.num_classes += num_new_classes
        old_output = self.output.cpu()
        self.output = nn.Conv2d(self.feature_size, self.num_anchors * self.num_classes, kernel_size=3, padding=1)

        # init output layer, this process is same as ResNet __init__()
        prior = 0.01 
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(0)
        # self.output.bias.data.fill_(-math.log((1.0 - prior) / prior))
        
        # copy old weight and bias    
        for i in range(self.num_anchors):
            self.output.weight.data[i * self.num_classes:i * self.num_classes + num_old_class,:,:,:] = old_output.weight.data[i * num_old_class:(i+1) * num_old_class,:,:,:] 
            self.output.bias.data[i * self.num_classes:i * self.num_classes + num_old_class] = old_output.bias.data[i * num_old_class:(i+1) * num_old_class]
        

        if method == "mean":
            # copy weight from the most similar class
            for new_class_id in range(num_new_classes):
                for old_class_id, ratio in enumerate(similaritys[new_class_id]):
                    for i in range(self.num_anchors):
                        self.output.weight.data[i * self.num_classes + num_old_class + new_class_id,:,:,:] += ratio * old_output.weight.data[i * num_old_class + old_class_id,:,:,:] 
                        self.output.bias.data[i * self.num_classes + num_old_class + new_class_id] += ratio * old_output.bias.data[i * num_old_class + old_class_id]
        #TODO 修改成複數個class
        elif method == "large":
            max_idx = int(torch.argmax(similaritys[0]))
            # copy weight from the most similar class
            for i in range(self.num_anchors):
                self.output.weight.data[i * self.num_classes + num_old_class,:,:,:] = old_output.weight.data[i * num_old_class + max_idx,:,:,:] 
                self.output.bias.data[i * self.num_classes + num_old_class] = old_output.bias.data[i * num_old_class + max_idx]
        else:
            for i in range(self.num_anchors):
                self.output.bias.data[i * self.num_classes + num_old_class: (i+1) * self.num_classes] = -math.log((1.0 - prior) / prior)
        self.output.cuda()
        del old_output


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers):
 
        self.num_classes = num_classes
  
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

        # self.focalLoss = losses.FocalLoss()

        # init weight and bias
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
    
    def freeze_layers(self, white_list:list):
        """freeze some layers 
            Args:
                white_list: list, indicate which layers don't be freezed
        """
        if white_list == None:
            return

        self.unfreeze_layers()
        debug_print("Freeze some layers, except", " and ".join(white_list))
        def keyword_check(name, white_list):
            if white_list == []:
                return True
            for word in white_list:
                if word in name:
                    return False
            return True
        
        for name, p in self.named_parameters():
            if keyword_check(name, white_list):
                p.requires_grad = False
            # if "bn" not in name:
                
        self.freeze_bn()

    def unfreeze_layers(self):
        """unfreeze all layers, except batch normalization layer
        """
        debug_print("Unfreeze all layers!")
        for name, p in self.named_parameters():
            p.requires_grad = True
            # if "bn" not in name:
                
        self.freeze_bn()
  

    def forward_feature(self, img_batch):
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features = self.fpn([x2, x3, x4])
        return features
        
    def forward(self, img_batch, return_feat=False, return_anchor=True, enable_act=True):
        """ model forward transfer
            Args:
                img_batch: tensor, shape = (batch_size, channel, height, width)
                return_feat: bool, whether return feature, default = False
                return_anchor: bool, whether return anchors, default = False
                enable_act: bool, whether enable classification subnet output layer activate, default = True

            Return: 
                tuple, value=(classification, regression, feature, anchors)
        """
        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        features = self.fpn([x2, x3, x4])
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)  #shape = (batch_size, W*H*A(Anchor_num), 4)
        
        classification = torch.cat([self.classificationModel(feature, enable_act) for feature in features], dim=1) #shape = (batch_size, W*H*A(Anchor_num), class_num)

        result = [classification, regression]
        if return_feat:
            result.append(features)
        if return_anchor:
            result.append(self.anchors(img_batch))
        
        return tuple(result)

    def cal_simple_focal_loss(self, img_batch, annotations, params):
        classification, regrsssion, anchors = self.forward(img_batch, return_feat=False, return_anchor=True, enable_act=True)
        loss = losses.FocalLoss().forward(classification, regrsssion, anchors, annotations, cur_state=0, params=params)

        cls_loss = loss['cls_loss'] 
        reg_loss = loss['reg_loss'] 
        return cls_loss, reg_loss

    def predict(self, img_batch, thresh=None, method=None):
        """ model prediction

            Args:
                img_batch: tensor, shape = (batch_size, channel, height, width)
                thresh: list, indicate each category's thresh
        """
        classification, regression , anchors = self.forward(img_batch, return_feat=False, return_anchor=True)
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

        # if thresh == None, then use default thresh
        if thresh == None:
            thresh = [0.05 for _ in range(classification.shape[2])]

        if len(thresh) != classification.shape[2]:
            raise ValueError("Parameter Thresh  must contain {} elements!".format(classification.shape[2]))

        thresh = 0.05
        scores = torch.squeeze(classification[0, :, :])
        anchorBoxes = torch.squeeze(transformed_anchors)


        scores, max_idxs = torch.max(scores, dim=1)
        scores_over_thresh = (scores > 0.05)
        scores = scores[scores_over_thresh]
        max_idxs = max_idxs[scores_over_thresh]    
        anchorBoxes = anchorBoxes[scores_over_thresh]
        anchors_nms_idx  = torchvision.ops.batched_nms(anchorBoxes, scores, max_idxs, 0.5)

        finalResult[0].extend(scores[anchors_nms_idx])
        finalResult[1].extend(max_idxs[anchors_nms_idx])
        finalResult[2].extend(anchorBoxes[anchors_nms_idx])

        finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
        finalAnchorBoxesIndexesValue = max_idxs[anchors_nms_idx].cuda()

        finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
        finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
        # Default predict method
#         if method == None:
#             for i in range(classification.shape[2]):
#                 scores = torch.squeeze(classification[:, :, i])
                
#                 scores_over_thresh = (scores > thresh[i]) # default thresh = 0.05

#                 # no boxes to NMS, just continue
#                 if scores_over_thresh.sum() == 0:
#                     continue
                    
#                 scores = scores[scores_over_thresh]
#                 anchorBoxes = torch.squeeze(transformed_anchors)
#                 anchorBoxes = anchorBoxes[scores_over_thresh]
                
#                 anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
                
#                 finalResult[0].extend(scores[anchors_nms_idx])
#                 finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
#                 finalResult[2].extend(anchorBoxes[anchors_nms_idx])

#                 finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
#                 finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
#                 if torch.cuda.is_available():
#                     finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()
#                 finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
#                 finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
        
#         # Experimental method
#         elif method == "large_score":
#             scores = torch.squeeze(classification[0, :, :])
#             temp = torch.max(scores, dim=1)
#             scores = temp[0]
#             max_idxs = temp[1]
            
#             scores_over_thresh = (scores > 0.05)

#             scores = scores[scores_over_thresh]
#             anchorBoxes = torch.squeeze(transformed_anchors)
#             anchorBoxes = anchorBoxes[scores_over_thresh]

#             anchors_nms_idx = nms(anchorBoxes, scores, 0.5)
            
            
#             finalResult[0].extend(scores[anchors_nms_idx])
#             finalResult[1].extend(max_idxs[anchors_nms_idx])
#             finalResult[2].extend(anchorBoxes[anchors_nms_idx])
            
#             finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
#             finalAnchorBoxesIndexesValue = max_idxs[anchors_nms_idx]
#             if torch.cuda.is_available():
#                 finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()
#             finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
#             finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
        return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]
    

    def next_state(self, num_new_classes:int, similarity, method="mean"):
        """next state
        Args:
            num_new_classes: the number of new classes number
        """
        debug_print("Add new neurons in classification Moddel output layers!")
        self.num_classes += num_new_classes
        self.classificationModel.next_state(num_new_classes, similarity, method)

def create_retinanet(depth:int, num_classes, pretrained=True, **kwargs):
    """Construct retinanet
        Args:
            depth: resnet's model depth
            num_classes: classification output layers's class num
            pretrained: whether resnet is pretrained, default = True
    """ 
    arch = {18:(BasicBlock,[2, 2, 2, 2]),
            34:(BasicBlock,[3, 4, 6, 3]),
            50:(Bottleneck,[3, 4, 6, 3]),
            101:(Bottleneck,[3, 4, 23, 3]),
            152:(Bottleneck,[3, 8, 36, 3])}

    if depth not in arch.keys():
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    model_arch = arch[depth]
    model = ResNet(num_classes, model_arch[0], model_arch[1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet{}'.format(depth)], model_dir='.'), strict=False)
    return model
    
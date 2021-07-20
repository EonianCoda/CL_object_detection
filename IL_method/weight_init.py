
import torch
from torchvision import transforms
from retinanet.dataloader import Resizer, Augmenter, Normalizer
THRESOLD = 0.5

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

def get_similarity(model, dataset_train, thresold=THRESOLD):
    new_class_num = len(dataset_train.seen_class_id)
    old_class_num = model.num_classes

    # not use Augmenter
    dataset_train.transform = transforms.Compose([Normalizer(), Resizer()])
    weight_similarity  =  Weight_similarity(model, new_class_num, old_class_num, thresold)

    img_count = torch.zeros(new_class_num, device=torch.device('cuda:0'))
    similaritys = torch.zeros(new_class_num, old_class_num, device=torch.device('cuda:0'))


    similarity = torch.zeros(new_class_num, old_class_num, device=torch.device('cuda:0')).float()
    label_count = torch.zeros(new_class_num, device=torch.device('cuda:0'))
    class_appear = torch.zeros(new_class_num, device=torch.device('cuda:0'))

    for idx, data in enumerate(dataset_train):
        
        with torch.no_grad():
            scores, labels = weight_similarity.forward(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0),
                                                data['annot'].cuda().unsqueeze(dim=0))
            if len(labels) == 0:
                continue

            labels = labels.long() - old_class_num

            similarity.fill_(0)
            label_count.fill_(0)
            class_appear.fill_(0)
            for idx, label in enumerate(labels):
                similarity[label] += scores[idx]
                label_count[label] += 1
                class_appear[label] = 1 
            similarity /= torch.clamp(label_count.unsqueeze(dim=1), min=1)

            similaritys += similarity
            img_count += class_appear[label]

    similaritys /= img_count.unsqueeze(dim=1)

    dataset_train.transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()])

    # discard very low category
    similaritys = torch.where(similaritys > 0.05, similaritys, torch.zeros(similaritys.shape, device=torch.device('cuda:0')))
    similaritys = similaritys / torch.sum(similaritys)
    similaritys = similaritys.cpu()
    return similaritys


class Weight_similarity(object):
    def __init__(self, model, new_class_num:int, old_class_num:int, thresold=THRESOLD):
        self.model = model
        self.new_class_num = new_class_num
        self.old_class_num = old_class_num
        self.thresold = thresold

    def forward(self, img_batch, annotations):
        classifications, _ , anchors = self.model(img_batch, 
                                                    return_feat=False, 
                                                    return_anchor=True, 
                                                    enable_act=True)
    
        classification = classifications[0, :, :]
        bbox_annotation = annotations[0, :, :]
        bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]
            
        classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

        if bbox_annotation.shape[0] == 0:  
            return
        

        IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # shape=(num_anchors, num_annotations)
        IoU_max, IoU_argmax = torch.max(IoU, dim=1) # shape=(num_anchors x 1)
        
        positive_indices = torch.ge(IoU_max, 0.5)
        
        greater = torch.ge(torch.sum(classification, dim = 1), self.thresold)
        
        indices = torch.logical_and(positive_indices, greater)
        
        classification = classification[indices,:]
        
        IoU_max = IoU_max[indices]

        classification = classification[:,:] / torch.sum(classification, dim = 1).unsqueeze(dim=1)
        # ground truth label
        assigned_annotations = bbox_annotation[IoU_argmax, :][indices,4]
        
        return classification, assigned_annotations

            

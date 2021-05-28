import argparse
import collections

import os
import numpy as np
import time
import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval
from retinanet.dataloader import CocoDataset_inOrder, rehearsal_DataSet
import pickle
import copy
assert torch.__version__.split('.')[0] == '1'

root_path = ''
method = 'w_distillation'
w_distillation = True
data_split = ""

# rehearsal
rehearsal_method = "random"
rehearsal_per_num = 2

# enhance new taks on old picture's loss
enhance_error = False

# when calculate new task'focal loss，the ground truth label 1 -> 0.9
decrease_positive = False

# warm-up
enable_warm_up = True
warm_up_epoch = 10

def autoDelete(model_path, now_round, now_epoch):
    global data_split
    path = os.path.join(model_path, 'round{}'.format(now_round), data_split)
    files = os.listdir(path)
    for i in range(1, now_epoch):
        checkpoint_path = os.path.join(path, 'voc_retinanet_{}_checkpoint.pt'.format(i))
        if i % 5 == 0:
            continue
        elif os.path.isfile(checkpoint_path):
            os.remove(checkpoint_path)

def checkDir(path):
    """check whether directory exists or not.If not, then create it 
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def get_checkpoint_path(method, now_round, epoch):
    global root_path
    global data_split
    
    checkDir(os.path.join(root_path, 'model', method, 'round{}'.format(now_round)))
    checkDir(os.path.join(root_path, 'model', method, 'round{}'.format(now_round), data_split))
    
    path = os.path.join(root_path, 'model', method, 'round{}'.format(now_round), data_split,'voc_retinanet_{}_checkpoint.pt'.format(epoch))
    return path

def readCheckpoint(method, now_round, epoch, retinanet, optimizer = None, scheduler = None):
    print('readcheckpoint at Round{} Epoch{}'.format(now_round, epoch))
    prev_checkpoint = torch.load(get_checkpoint_path(method, now_round, epoch))
    retinanet.load_state_dict(prev_checkpoint['model_state_dict'])
    if optimizer != None:
        optimizer.load_state_dict(prev_checkpoint['optimizer_state_dict'])
    if scheduler != None:
        scheduler.load_state_dict(prev_checkpoint['scheduler_state_dict'])

def load_previous_model(now_round, classOrder):
    print('Load previous model on Round{}'.format(now_round))
    # classOrder = pickle.load(open(os.path.join(root_path, 'DataSet', 'VOC2012', 'annotations', 'instances_TrainVoc2012_classOrder.pickle'), 'rb'))

    num_classes = 0
    for i in range(now_round):
        num_classes += len(classOrder['id'][i])
    prev_model = model.resnet50(num_classes=num_classes, pretrained=True)
    print(num_classes)
    readCheckpoint(method, now_round , 50, prev_model)
    del classOrder
    return prev_model
    
def validation(val_model, set_name, model_round, model_epoch, val_round):
    global data_split
    print("-"*100)
    print('Start eval on Round{} Epoch{}!'.format(model_round, model_epoch))

    
    val_model.eval()
    val_model.freeze_bn()
    #set_name = "{}Voc2012".format(dataType, )
    if "2012" in set_name:
        years = "VOC2012"
    else:
        years = "VOC2007"
    
    print('Validation data is {} at Round{}'.format(set_name, val_round))
    dataset_val = CocoDataset_inOrder(os.path.join(root_path, 'DataSet', years), set_name=set_name, dataset = 'voc', 
                    transform=transforms.Compose([Normalizer(), Resizer()]), 
                    start_round=val_round, data_split = data_split)
 
    coco_eval.evaluate_coco(dataset_val, val_model, root_path, method, model_round, model_epoch)
    del dataset_val

print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
    global root_path
    global method
    global w_distillation
    global data_split
    global rehearsal_method
    global rehearsal_per_num
    global enhance_error
    global decrease_positive
    global enable_warm_up
    global warm_up_epoch
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--batch_size', help='batch_size', type=int, default=5)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epoch', help='Number of epochs', type=int, default=50)
    parser.add_argument('--start_round', type=int, default=1)
    parser.add_argument('--total_round', type=int, default=0)
    parser.add_argument('--val',help='whether do validation', default='None')
    parser.add_argument('--val_round', type=int, default=0)
    parser.add_argument('--method', default='w_distillation')
    parser.add_argument('--split')
    parser = parser.parse_args(args)


    root_path = '/'.join(parser.coco_path.split('/')[:parser.coco_path.split('/').index('DataSet')])
    method = parser.method
    data_split = parser.split
    print('data split is ', data_split)
    if method != 'w_distillation':
        w_distillation = False
    print('Use method:', method)
    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'voc':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset_inOrder(parser.coco_path, set_name='TrainVoc2012', dataset = 'voc',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]), 
                                    start_round=parser.start_round, data_split = data_split)
        if rehearsal_method != None:
            img_ids = None
            #good example
            img_ids = [2010005199,2009000248,2009000777,2008002204,2008006064,2009004121,2011000804,2008005761,2010005182,
                     2008006898,2009005031,2009001390,2009002343,2009003482,2009000604,2010005266,2009002060,
                       2010000492,2008003629,2011002422,2008001856,2009003685,2008002872,2010004808,
                       2008005714,2008006355,2010002696,2009000088,2010001503,2008007237]
            #bad example
#             img_ids = [2009004162,2011000420,2010001550,2008008701,2010005670,2008007421,2009004871,
#                          2009004328,2011000651,2008004365,2008008479,2008007629,2008006586,2008007653,
#                          2011000492,2008005616,2008007697,2009004984,2010001366,2009000969,2009001693,2009000503,
#                          2010002529,2011001971,2011002189,2009002472,2009000774,2009004683,2011002884,2009002416]
            if img_ids == None:
                rehearsal_dataset = rehearsal_DataSet(parser.coco_path, set_name='TrainVoc2012', dataset = 'voc',
                                        transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]), 
                                        data_split = data_split,method = rehearsal_method, per_num = rehearsal_per_num)
            else:
                rehearsal_dataset = rehearsal_DataSet(parser.coco_path, set_name='TrainVoc2012', dataset = 'voc',
                                        transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]), 
                                        data_split = data_split,method = rehearsal_method, per_num = rehearsal_per_num,img_ids=img_ids)
            dataloader_rehearsal = None
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size = parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    
    use_gpu = True
    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    
    #very special increase
    ###################################################################################
#     readCheckpoint(method, 0, 20, retinanet, optimizer, scheduler)
#     retinanet.special_increase(dataset_train)
#     optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
#     dataset_train.num_classes()
    ####################################################################################
    
    #read checkpoint
    if parser.start_round != 1 or parser.start_epoch != 1:
        if parser.start_round != 1 and parser.start_epoch == 1:
            print('read checkpoint Error')
            return
        
        readCheckpoint(method, parser.start_round, parser.start_epoch - 1, retinanet, optimizer, scheduler)
        if parser.start_round > 1 and method == 'w_distillation' and parser.val == 'None':
            retinanet.init_prev_model(load_previous_model(parser.start_round - 1, dataset_train.classOrder))
        
            if rehearsal_method != None:
                print('read hearsal dataset chekcpoint at Round{},Epoch{}'.format(parser.start_round, parser.start_epoch - 1))
                checkpoint = torch.load(get_checkpoint_path(method, parser.start_round, parser.start_epoch - 1))
                
                rehearsal_dataset.reset_by_imgIds(checkpoint['rehearsal_per_num'], checkpoint['rehearsal_samples'])
                del checkpoint
                
                    
    #validation
    if parser.val != 'None':
        if parser.val_round == 0:
            parser.val_round = parser.start_round
        validation(retinanet, parser.val, parser.start_round, parser.start_epoch - 1, parser.val_round)
        return
    
    

    prev_model = None
    for now_round in range(parser.start_round, parser.total_round + 1):
        print("-----------------------------------------------------------------")
        print("Round {}:".format(now_round))
        print('Num training images: {}'.format(len(dataset_train)))
        print('Iteration_num: ',len(dataloader_train))

        if parser.start_epoch == parser.total_epoch:
            epoch_num = parser.start_epoch

        #whem next round, reset start epoch
        if now_round != parser.start_round:
            parser.start_epoch = 1
            
        
        # Freeze model, do warm-up
#         if enable_warm_up and now_round != 2:
#             retinanet.freeze_except_new_classification(False)
        if enable_warm_up:
            retinanet.freeze_resnet(False)
          
        retinanet.train()
        retinanet.freeze_bn()
        
        for epoch_num in range(parser.start_epoch , parser.total_epoch + 1):
            
            # unfreeze model, stop warm-up
            if epoch_num == warm_up_epoch + 1: 
#                 retinanet.freeze_except_new_classification(True)
#                 retinanet.freeze_bn()
                retinanet.freeze_resnet(True)
                retinanet.freeze_bn()
            
            
            # Warm-up Hint
            if enable_warm_up and epoch_num <= warm_up_epoch:
                print('Warm up')
            epoch_loss = []
            
#             if epoch_num > 5:
#                 retinanet.set_special_alpha(10.0)
            retinanet.decrease_positive = decrease_positive
            for iter_num, data in enumerate(dataloader_train):
                start = time.time()
                try:
                    optimizer.zero_grad()
                    with torch.cuda.device(0):
                        if torch.cuda.is_available():
                            losses = retinanet([data['img'].float().cuda(), data['annot'].cuda()])
                            if retinanet.distill_loss:
                                classification_loss, regression_loss, dist_class_loss, dist_reg_loss, dist_feat_loss = losses
                            else:
                                classification_loss, regression_loss = losses
                        else:
                            print('not have gpu')
                            return

                        classification_loss = classification_loss.mean()
                        regression_loss = regression_loss.mean()
                        loss = classification_loss + regression_loss

                        #add distillation loss
                        if retinanet.distill_loss:
                            loss += dist_class_loss + dist_reg_loss + dist_feat_loss

                        if bool(loss == 0):
                            continue

                        loss.backward()


                        #Eliminate old task grad
                        if enable_warm_up and now_round > 1 and epoch_num <= warm_up_epoch:
                            for i in range(retinanet.classificationModel.num_anchors):
                                retinanet.classificationModel.output.weight.grad[i * retinanet.num_classes : i * retinanet.num_classes + retinanet.prev_num_classes,:,:,:] = 0
                                retinanet.classificationModel.output.bias.grad[i * retinanet.num_classes : i * retinanet.num_classes + retinanet.prev_num_classes] =  0
                        torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)



                        optimizer.step()
                        loss_hist.append(float(loss))

                        epoch_loss.append(float(loss))
                        end = time.time()

                        if not retinanet.distill_loss:
                            print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f} | Spend Time:{:1.2f}s'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist),end - start))
                        else:
                            print('Epoch: {} | Iteration: {} | Class loss: {:1.4f} | Reg loss: {:1.4f} | dis_class_loss: {:1.4f} | dis_reg_loss: {:1.4f} | dis_feat_loss: {:1.4f} | Running loss: {:1.5f} | Spend Time:{:1.2f}s'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), float(dist_class_loss), float(dist_reg_loss), float(dist_feat_loss), np.mean(loss_hist),end - start))

                        del classification_loss
                        del regression_loss
                        if retinanet.distill_loss:
                            del dist_class_loss, dist_reg_loss, dist_feat_loss
                except Exception as e:
                    print(e)
                    continue
            
            retinanet.decrease_positive = False
            if rehearsal_method != None and now_round != 1:

                if rehearsal_dataset.now_round == None:
                    if len(rehearsal_dataset.image_ids) == 0:
                        rehearsal_dataset.reset_by_round(now_round)
                    else:
                        rehearsal_dataset.now_round = now_round
                
                if dataloader_rehearsal == None:
                    sampler = AspectRatioBasedSampler(rehearsal_dataset, batch_size = 3, drop_last=False)
                    dataloader_rehearsal = DataLoader(rehearsal_dataset, num_workers=2, collate_fn=collater, batch_sampler=sampler)  
                prev_status = retinanet.distill_loss
                retinanet.distill_loss = False
                
                retinanet.enhance_error = enhance_error
                print('Rehearsal start!')
                
                
                for iter_num, data in enumerate(dataloader_rehearsal):
                    if enable_warm_up and epoch_num <= warm_up_epoch:
                        break
                    start = time.time()
                    try:
                        optimizer.zero_grad()
                        with torch.cuda.device(0):
                            if torch.cuda.is_available():
                                losses = retinanet([data['img'].float().cuda(), data['annot'].cuda()])
                                classification_loss, regression_loss = losses
                            else:
                                print('not have gpu')
                                return

                            classification_loss = classification_loss.mean()
                            regression_loss = regression_loss.mean()
                            loss = classification_loss + regression_loss

                            if bool(loss == 0):
                                continue

                            loss.backward()

                            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                            optimizer.step()
                            loss_hist.append(float(loss))

                            epoch_loss.append(float(loss))
                            end = time.time()

                            print('Rehearsal Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f} | Spend Time:{:1.2f}s'.format(epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist),end - start))
                            
                            del classification_loss
                            del regression_loss
                    except Exception as e:
                        print(e)
                        continue
                
            
                retinanet.distill_loss = prev_status
                retinanet.enhance_error = False

            scheduler.step(np.mean(epoch_loss))
            savePath = get_checkpoint_path(method, now_round, epoch_num)
            #if thd directory doesn't exist, then create it
            dirPath = '/'.join(savePath.split('/')[:-1])
            checkDir(dirPath)
            
            #save checkpoint
            if retinanet.prev_model != None:
                prev_model = retinanet.prev_model
                retinanet.prev_model = None
                
                
            
            if rehearsal_method == None:
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': retinanet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': epoch_loss,
                    }, savePath)
            else:
                torch.save({
                    'epoch': epoch_num,
                    'model_state_dict': retinanet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': epoch_loss,
                    'rehearsal_samples': rehearsal_dataset.image_ids,
                    'rehearsal_per_num': rehearsal_dataset.per_num
                    }, savePath)
            
            
            retinanet.prev_model = prev_model
            if epoch_num % 5 == 0:
                autoDelete(os.path.join(root_path, 'model', method), now_round, epoch_num)
        

        if now_round == parser.total_round:
            #validation(retinanet, 'Val', 2, 50, 1)
            break
        
        #add new neuron on the classification subnet
        origin_num = dataset_train.num_classes()
        dataset_train.next_round()
        new_class_num = dataset_train.num_classes() - origin_num
        
        retinanet.increase_class(new_class_num, w_distillation)
        print('new classes num is {}'.format(new_class_num))

        #reset optimizer and scheduler
        prev_model = retinanet.prev_model
        retinanet.prev_model = None
        
        optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
        retinanet.prev_model = prev_model
        
        if rehearsal_method != None and rehearsal_dataset.now_round != None and ((now_round == 1) ^ (len(rehearsal_dataset.image_ids) != 0)):
            rehearsal_dataset.next_round()
            sampler = AspectRatioBasedSampler(rehearsal_dataset, batch_size = 3, drop_last=False)
            dataloader_rehearsal = DataLoader(rehearsal_dataset, num_workers=2, collate_fn=collater, batch_sampler=sampler)
       
            
        #change the training data
        sampler = AspectRatioBasedSampler(dataset_train, batch_size = parser.batch_size, drop_last=False)
        dataloader_train = DataLoader(dataset_train, num_workers=2, collate_fn=collater, batch_sampler=sampler)
        
if __name__ == '__main__':
    main()

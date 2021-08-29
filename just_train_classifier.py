import argparse
import collections

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
    # Training dataloader
    dataset_train = IL_dataset(params,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
                               start_state=start_state,
                               use_all_class=True)

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
    
    # IL_Trainer

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

def main(args=None):
    parser = get_parser(args)
    params = Params(parser)

    model, optimizer, dataloader_train = create_tools(params)

    model.train()
    model.freeze_layers(['classificationModel.output','regressionModel.output'])
    loss_hist = collections.deque(maxlen=500)
    print("Total IterNum:",len(dataloader_train))
    for epoch in range(params['start_epoch'], params['end_epoch'] + 1):
        for iter_num, data in enumerate(dataloader_train):
            start = time.time()
            fast_zero_grad(model)

            with torch.cuda.device(0):
                try:
                    cls_loss, reg_loss = model.cal_simple_focal_loss(data['img'].float().cuda(), 
                                                data['annot'].cuda(),
                                                params)

                    # if cls_loss < 0.005:
                    #     continue
                    
                    loss = cls_loss + reg_loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()
                    loss_hist.append(float(loss))
                    end = time.time()
                    print("Epoch: {} | Iter: {} | Cls_loss: {:.3f} | Reg_loss: {:.3f} | Total_loss: {:.3f} | Running_loss: {:.3f} | Time: {:.2f}s".format(epoch, 
                                                                                                                                                    iter_num, 
                                                                                                                                                    float(cls_loss), 
                                                                                                                                                    float(reg_loss), 
                                                                                                                                                    float(loss),
                                                                                                                                                    np.mean(loss_hist),
                                                                                                                                                    float(end - start)))
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


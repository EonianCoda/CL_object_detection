import argparse
import collections
# torch 
import torch
import torch.optim as optim
from torchvision import transforms
# retinanet
from retinanet.model import create_retinanet
from retinanet.dataloader import IL_dataset
from retinanet.dataloader import Resizer, Augmenter, Normalizer
# preprocessing
from preprocessing.params import Params
# train
from train.il_trainer import IL_Trainer
from train.train import train_process
# Global Setting

ROOT_DIR = "/home/deeplab307/Documents/Anaconda/Shiang/IL/" 
PRINT_INFO = True # whether print some information about continual learning on the scrren
DEFAULT_ALPHA = 0.25
DEFAULT_GAMMA = 2.0

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def create_IL_trainer(params:Params):
    start_state = params['start_state']
    start_epoch = params['start_epoch']
    # Training dataloader
    dataset_train = IL_dataset(params,
                               transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
                               start_state=start_state,
                               use_data_ratio = params['use_data_ratio'])

    # Create the model
    if start_epoch == 1 and start_state != 0:
        retinanet = create_retinanet(params['depth'], params.states[start_state - 1]['num_knowing_class'])
    else:
        retinanet = create_retinanet(params['depth'], params.states[start_state]['num_knowing_class'])
    retinanet = retinanet.cuda()
    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    
    # Read checkpoint
    if start_state != 0 or start_epoch != 1:
        if start_epoch == 1:
            params.load_model(start_state - 1,-1,retinanet, optimizer, scheduler, loss_hist)
        else:
            params.load_model(start_state,start_epoch - 1,retinanet, optimizer, scheduler, loss_hist)
    
    # IL_Trainer
    trainer = IL_Trainer(params,
                            model=retinanet,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            dataset_train=dataset_train,
                            loss_hist=loss_hist)
    if start_epoch == 1 and start_state != 0:
        trainer.update_training_tools()
    return trainer

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
    # IL params 
    parser.add_argument('--scenario', help='the scenario of states, must be "20", "19 1", "10 10", "15 1", "15 1 1 1 1"', type=int, nargs="+", default=[20])
    parser.add_argument('--suffle_class', help='whether shuffle the class, default = False',type=str2bool , default=False)

    parser.add_argument('--distill', help='whether add distillation loss, default = False',type=str2bool , default=False)
    parser.add_argument('--distill_logits', help='whether distillation loss use logits, default = False',type=str2bool , default=False)
    parser.add_argument('--distill_logits_on', help='whether distillation loss use logits on new class or old class,two option:"new" or ""old default = new', default="new")
    parser.add_argument('--distill_logits_bg_loss', help='whether add background loss on distillation loss, default = False',type=str2bool , default=False)

    parser.add_argument('--sample_num', help='the number of sample images each class for replay metohd, 0 mean no sample, default = 0', type=int, default=0)
    parser.add_argument('--sample_method', help="sample old state images's method, must be 'random','large_loss','custom'", default="custom")

    parser.add_argument('--mas', help='whether add memory aware synapses loss, must be "true" or "false", default="false"',type=str2bool , default=False)
    parser.add_argument('--mas_file', help='the name of mas file name, default="MAS"', default="MAS")

    parser.add_argument('--agem', help='whether add averaged gradient episodic memory loss, must be "true" or "false", default="false"',type=str2bool , default=False)
    # parser.add_argument('--agem_batch', help='the number of agem batch size use , -1 mean use all category', type=int, default=-1)

    parser.add_argument('--warm_stage', help='the number of warm-up stage, 0 mean not warm up, default = 0', type=int, default=0)
    parser.add_argument('--warm_epoch', help='the number of epoch for each warm-up stage, use " "(space) to split epoch for different stage', type=int, nargs='*', default=[10,10])
    parser.add_argument('--warm_layers', help='the layers which will be warmed up, must be "output", "resnet", "fpn", and split each stage by space " "', nargs='*', default=['output','resnet'])

    # IL experimental params
    parser.add_argument('--ignore_ground_truth', help='whether ignore ground truth, default = False',type=str2bool , default=False)
    parser.add_argument('--decrease_positive', help="the upper score of the new class in incremental state, default=1.0",type=float , default=1.0) 
    parser.add_argument('--enhance_error', help="when use naive replay method, whether enhance new task error or not",type=str2bool , default=False) 
    parser.add_argument('--enhance_error_method', help='if enhance new task error, which method to use, must be "L1","L2","L3"', default="L2") 


    # always default paras
    parser.add_argument('--print_il_info', help='whether debug in Train process, default = False', type=str2bool, default=False)
    parser.add_argument('--debug', help='whether debug in Train process, default = False', type=str2bool, default=False)
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=5)
    parser.add_argument('--new_state_epoch', help='the number of new state training epoch', type=int, default=60)
    parser.add_argument('--use_data_ratio', type=int, default=1)
    parser.add_argument('--ignore_past_class', help='when calculating the focal loss, whether ignore past class), default = True',type=str2bool , default=True)
    parser = vars(parser.parse_args(args))
    return parser

def main(args=None):
    parser = get_parser(args)
    params = Params(parser)
    params.output_params()
    il_trainer = create_IL_trainer(params)
    # print training information
    if PRINT_INFO:
        print("Scenario:", params['scenario'])
        print("State from {} to {}".format(params['start_state'], params['end_state']))
        print("States Information:".format(params['start_state'], params['end_state']))
        print('-'*70)
        params.states.print_state_info()
        print('-'*70)
        print("Start Training!")
        print('-'*70)

    if params['print_il_info']:
        params.print_il_info()
    train_process(il_trainer)

        
if __name__ == '__main__':
    assert torch.__version__.split('.')[0] == '1'
    if not torch.cuda.is_available():
        print("CUDA isn't abailable")
        exit(-1)
    main()


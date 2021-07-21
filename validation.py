# built-in
import argparse
import os
import shutil
# torch
import torch
from evaluator import Evaluator, multi_evaluation


ROOT_DIR = "/home/deeplab307/Documents/Anaconda/Shiang/IL/"
DEFAULT_THRESHOLD = 0.05
DEFAULT_DEPTH = 50

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_val_parser(args=None):
    parser = argparse.ArgumentParser()
    # must set params
    
    parser.add_argument('--dataset', help='Dataset name, must contain name and years, for instance: voc2007,voc2012', default='voc2007')
    parser.add_argument('--epoch', help='the index of validation epochs', nargs='+', type=int)
    parser.add_argument('--state', type=int)
    parser.add_argument('--scenario', help='the scenario of states, must be "20", "19 1", "10 10", "15 1", "15 1 1 1 1"', type=int, nargs="+", default=[20])
    parser.add_argument('--threshold', help='the threshold for prediction default=0.05', type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument('--just_val', help='whether predict or not',type=str2bool, default=False)
    parser.add_argument('--output_csv', help='whether output the csv file, default = True', type=str2bool, default=True)
    parser.add_argument('--timestamp',help='whether create new folder in val_result, default = True',type=str2bool, default=True)

    # always fixed
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=DEFAULT_DEPTH)
    parser.add_argument('--root_dir', help='the root dir for training', default=ROOT_DIR)
    parser = vars(parser.parse_args(args))
    # set for origin Parmas, otherwise it will have error
    parser['warm_stage'] = 0
    parser['shuffle_class'] = False
    return parser


def validation(evaluator:Evaluator):
    epochs = list(set(evaluator['epoch']))

    # copy params.txt
    ckp_path = os.path.join(evaluator['ckp_path'], 'state{}'.format(evaluator['state']))
    params_file = os.path.join(ckp_path, 'params.txt')
    if os.path.isfile(params_file):
        shutil.copy(params_file, os.path.join(evaluator.get_result_path(-1), 'params.txt'))
    
    print("Evaluate at state{} Epoch({})".format(evaluator['state'], evaluator['epoch']))

    if evaluator['just_val']:
        evaluator.validation_check(epochs)
        for epoch in epochs:
            evaluator.do_evaluation(epoch)
    else:
        multi_evaluation(evaluator, epochs)


    if evaluator['timestamp']:
        from torch.utils.tensorboard import SummaryWriter


            
        logdir = os.path.join(evaluator.get_result_path(-1),'runs', evaluator.new_folder_name)
        with SummaryWriter(logdir) as w:
            hparams = evaluator.get_il_info()
            eval_results = evaluator.get_tensorbord_info()

            for epoch in eval_results.keys():
                hparams['epoch'] = epoch

                w.add_hparams(hparams,
                              eval_results[epoch])

    if evaluator['output_csv']:
        evaluator.output_csv_file()

def main(args=None):
    parser = get_val_parser(args)
    evaluator = Evaluator(parser)
    validation(evaluator)
    

if __name__ == '__main__':
    assert torch.__version__.split('.')[0] == '1'
    if not torch.cuda.is_available():
        print("CUDA isn't abailable")
        exit(-1)
    main()

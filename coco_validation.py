import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer
from retinanet import coco_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

# def validation(val_model, set_name, model_round, model_epoch, val_round):
#     global data_split
#     print("-"*100)
#     print('Start eval on Round{} Epoch{}!'.format(model_round, model_epoch))

    
#     val_model.eval()
#     val_model.freeze_bn()
#     #set_name = "{}Voc2012".format(dataType, )
#     if "2012" in set_name:
#         years = "VOC2012"
#     else:
#         years = "VOC2007"
    
#     print('Validation data is {} at Round{}'.format(set_name, val_round))
#     dataset_val = CocoDataset_inOrder(os.path.join(root_dir, 'DataSet', years), set_name=set_name, dataset = 'voc', 
#                     transform=transforms.Compose([Normalizer(), Resizer()]), 
#                     start_round=val_round, data_split = data_split)
 
#     coco_eval.evaluate_coco(dataset_val, val_model, root_dir, method, model_round, model_epoch)
#     del dataset_val
def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer()]))

    # Create the model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    coco_eval.evaluate_coco(dataset_val, retinanet)


if __name__ == '__main__':
    main()

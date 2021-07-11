import torch
import time
import numpy as np


from retinanet.losses import IL_Loss

from train.il_trainer import IL_Trainer

IGNORE_BUG = False

def fast_zero_grad(model):
    for param in model.parameters():
        param.grad = None

def train_iter(il_trainer:IL_Trainer, il_loss:IL_Loss,data):
    """
        Args:
        Return: a dict, containing loss information
    """
    with torch.cuda.amp.autocast():

        losses = il_loss.forward(data['img'].float().cuda(), data['annot'].cuda())

        loss = torch.tensor(0).float().cuda()
        loss_info = {}
        for key, value in losses.items():
            loss += value
            loss_info[key] = float(value)

        if bool(loss == 0):
            return None
        loss.backward()

    
        torch.nn.utils.clip_grad_norm_(il_trainer.model.parameters(), 0.1)

        
        il_trainer.optimizer.step()
        il_trainer.loss_hist.append(float(loss))
        loss_info['total_loss'] = float(loss)

        for key, value in losses.items():
            del value


    return loss_info

def train_process(il_trainer : IL_Trainer):
    params = il_trainer.params

    for cur_state in range(params['start_state'], params['end_state'] + 1):
        print("State: {}".format(cur_state))
        print("Train epoch from {} to {}".format(params['start_epoch'], params['end_epoch']))
        print('Num training images: {}'.format(len(il_trainer.dataset_train)))
        print('Iteration_num: ',len(il_trainer.dataloader_train))

        # when next round, reset start epoch
        if cur_state != params['start_state']:
            start_epoch = 1
            end_epoch = params['new_state_epoch']
        else:
            start_epoch = params['start_epoch']
            end_epoch = params['end_epoch']
        
        for epoch in range(start_epoch, end_epoch):
            epoch_loss = []
            il_trainer.model.train()
            il_trainer.warm_up(epoch=epoch)
            il_trainer.model.freeze_bn()
            for iter_num, data in enumerate(il_trainer.dataloader_train):
                # if enable_warm_up and epoch_num >= warm_up_epoch + 1 and enable_agem:
                #     agem.cal_replay_grad(optimizer)
                    
                start = time.time()
                fast_zero_grad(il_trainer.model)
                if IGNORE_BUG:
                    try:
                        losses = train_iter(il_trainer, data)
                    except Exception as e:
                        print(e)
                        continue
                else:
                    losses = train_iter(il_trainer, data)

                if losses == None:
                    continue
                
                # Print Iteration Information
                info = [epoch, iter_num]
                output = 'Epoch: {0[0]} | Iteration: {0[1]}'
                for key, value in losses:
                    output += ' | {0[%d]}: {0[%d]:1.4f}' % (len(info), len(info+1))
                    info.extend([key, value])
                
                output += ' | Running loss: {0[%d]:1.5f} | Spend Time:{0[%d]:1.2f}s' % (len(info), len(info)+1)
                end = time.time()
                info.extend([np.mean(il_trainer.loss_hist), end - start])
                print(output.format(info))

                # Log epoch loss
                epoch_loss.append(losses['total_loss'])

            il_trainer.save_ckp(epoch_loss, epoch=epoch)
        il_trainer.next_state()
        
        

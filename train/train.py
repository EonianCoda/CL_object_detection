#build-in
import torch
import time
import numpy as np
# retinanet
from retinanet.losses import IL_Loss
from train.il_trainer import IL_Trainer
# tool
from recorder import Recorder

def fast_zero_grad(model):
    for param in model.parameters():
        param.grad = None

def training_iteration(il_trainer:IL_Trainer, il_loss:IL_Loss, data, is_replay=False):
    """
        Args:
        Return: a dict, containing loss information
    """
    # with torch.cuda.amp.autocast():
    with torch.cuda.device(0):
        losses = il_loss.forward(data['img'].float().cuda(), data['annot'].cuda(), is_replay=is_replay)

        loss = torch.tensor(0).float().cuda()
        loss_info = {}
        for key, value in losses.items():
            loss += value
            if is_replay:
                key = 'replay_' + key
            loss_info[key] = float(value)

        if bool(loss == 0):
            return None
        loss.backward()

        torch.nn.utils.clip_grad_norm_(il_trainer.model.parameters(), 0.1)
        
        if il_trainer.params['agem']:
            il_trainer.agem.fix_grad(il_trainer.model)

        il_trainer.optimizer.step()
        il_trainer.loss_hist.append(float(loss))
        loss_info['total_loss'] = float(loss)

        del losses
    return loss_info

def print_iteration_info(il_trainer, losses, cur_epoch:int, iter_num:int, spend_time:float):
    """Print Iteration Information

    """
    info = [cur_epoch, iter_num]
    output = 'Epoch: {0[0]:2d} | Iter: {0[1]:3d}'
    for key, value in losses.items():
        output += ' | {0[%d]}: {0[%d]:1.4f}' % (len(info), len(info)+1)
        info.extend([key, value])
    
    output += ' | Running loss: {0[%d]:1.5f} | Spend Time:{0[%d]:1.2f}s' % (len(info), len(info)+1)
    
    info.extend([np.mean(il_trainer.loss_hist), spend_time])
    print(output.format(info))

def cal_losses(il_trainer, il_loss, data, is_replay=False):
    fast_zero_grad(il_trainer.model)
    if not il_trainer.params['debug']:
        try:
            losses = training_iteration(il_trainer,il_loss, data, is_replay)
        except Exception as e:
            print(e)
            return None
    else:
        losses = training_iteration(il_trainer,il_loss, data, is_replay)

    if losses == None:
        return None
    return losses

def train_process(il_trainer : IL_Trainer):
    # init training info
    start_state = il_trainer.params['start_state']
    end_state = il_trainer.params['end_state']
    start_epoch = il_trainer.params['start_epoch']
    end_epoch = il_trainer.params['end_epoch']
    # Init Recorder
    recorder = Recorder(il_trainer)

    if end_state < start_state:
        end_state = start_state

    # init IL loss
    il_loss = IL_Loss(il_trainer)

    for cur_state in range(start_state, end_state  + 1):
        print("State: {}".format(cur_state))
        print("Train epoch from {} to {}".format(start_epoch, end_epoch))
        print('Num training images: {}'.format(len(il_trainer.dataset_train)))
        print('Iteration_num: ',len(il_trainer.dataloader_train))

        # when next round, reset start epoch
        if cur_state != start_state:
            start_epoch = 1
            end_epoch = il_trainer.params.params['new_state_epoch']
        
        for cur_epoch in range(start_epoch, end_epoch + 1):
            # Some Log 
            avg_times = []
            epoch_loss = []

            # Model setting
            il_trainer.model.train()
            il_trainer.warm_up(epoch=cur_epoch)
            il_trainer.model.freeze_bn()

            not_warm_output = not (il_trainer.cur_warm_stage != -1 and il_trainer.params['warm_layers'][il_trainer.cur_warm_stage] == 'output')

            # Training Dataset
            for iter_num, data in enumerate(il_trainer.dataloader_train):
                start = time.time()
                if il_trainer.params['agem']:
                    il_trainer.agem.cal_replay_grad(il_loss)
                    
                
                losses = cal_losses(il_trainer, il_loss, data)
                if losses == None:
                    continue
                end = time.time()
                print_iteration_info(il_trainer, losses, cur_epoch, iter_num, end - start)

                # Iteration Log
                epoch_loss.append(losses['total_loss'])
                avg_times.append(end - start)
        
                recorder.add_iter_loss(losses)


            # Replay Dataset
            if il_trainer.params['agem'] == False and il_trainer.dataset_replay != None and not_warm_output:
                print("Start Replay!")
                print('Num Replay images: {}'.format(len(il_trainer.dataset_replay)))
                print('Iteration_num: ',len(il_trainer.dataloader_replay))
                for iter_num, data in enumerate(il_trainer.dataloader_replay):
                    start = time.time()
                    losses = cal_losses(il_trainer, il_loss, data, is_replay=True)
                    if losses == None:
                        continue
                    end = time.time()
                    print_iteration_info(il_trainer, losses, cur_epoch, iter_num, end - start)

                # Iteration Log
                epoch_loss.append(losses['total_loss'])
                avg_times.append(end - start)
                recorder.add_iter_loss(losses)

            il_trainer.scheduler.step(np.mean(epoch_loss))
            il_trainer.save_ckp(epoch_loss, epoch=cur_epoch)
            il_trainer.params.auto_delete(cur_state, cur_epoch)

            # Epoch Log
            recorder.record_epoch_loss(cur_epoch)

            # Compute remaining training time
            avg_times = sum(avg_times)
            avg_times = avg_times * (end_epoch - cur_epoch)
            avg_times = (int(avg_times / 60), int(avg_times) % 60)
            print("Estimated Training Time for this state is {}m{}s".format(avg_times[0],avg_times[1]))

        
        if cur_state != end_state:
            il_trainer.next_state()
            if il_trainer.params['record']:
                recorder.next_state()
    recorder.end_write()
        
        

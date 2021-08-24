#build-in
import torch
import time
import numpy as np
# retinanet
from retinanet.losses import IL_Loss
from train.il_trainer import IL_Trainer
# tool
from recorder import Recorder
import random
def fast_zero_grad(model):
    for param in model.parameters():
        param.grad = None

def training_iteration(il_trainer:IL_Trainer, il_loss:IL_Loss, data, is_replay=False):
    """
        Args:
        Return: a dict, containing loss information
    """
    # with torch.cuda.amp.autocast():

    warm_classifier = (il_trainer.cur_warm_stage != -1) and (il_trainer.params['warm_layers'][il_trainer.cur_warm_stage] == 'output')

    with torch.cuda.device(0):
        losses = il_loss.forward(data['img'].float().cuda(), data['annot'].cuda(), is_replay=is_replay)

        loss = torch.tensor(0).float().cuda()
        loss_info = {}
        for key, value in losses.items():
            if value != None:
                loss += value
                if is_replay:
                    key = 'replay_' + key
                loss_info[key] = float(value)
            else:
                loss_info[key] = float(0)

        if bool(loss == 0):
            return None
        
        # mas penalty
        if not is_replay and il_trainer.params['mas']:
            mas_loss = il_trainer.mas.penalty(il_trainer.prev_model, il_trainer.params['mas_ratio'])
            loss_info['mas_loss'] = float(mas_loss)
            loss += mas_loss

        loss.backward()
        if not warm_classifier and not il_trainer.params['no_clip']:
            torch.nn.utils.clip_grad_norm_(il_trainer.model.parameters(), 0.1)

        # warm classifier
        if warm_classifier:
            classificationModel = il_trainer.model.classificationModel
            cur_state = il_trainer.cur_state
            num_classes = il_trainer.params.states[cur_state]['num_knowing_class']
            num_old_classes = il_trainer.params.states[cur_state]['num_past_class']
            for i in range(classificationModel.num_anchors):
                start_idx = i *  num_classes
                classificationModel.output.weight.grad[start_idx : start_idx + num_old_classes,:,:,:] = 0
                classificationModel.output.bias.grad[start_idx : start_idx + num_old_classes] = 0
        #Agem fix gradient
        if not is_replay and il_trainer.params['agem']:
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

            not_warm_classifier = not (il_trainer.cur_warm_stage != -1 and il_trainer.params['warm_layers'][il_trainer.cur_warm_stage] == 'output')



            num_training_iter = len(il_trainer.dataloader_train)

            replay_exist =  (il_trainer.params['agem'] == False) and (il_trainer.dataset_replay != None)
            if replay_exist:
                num_replay_iter = len(il_trainer.dataloader_replay)
                
                if num_replay_iter <= num_training_iter:
                    do_replay_ids = random.sample(range(num_training_iter), k=num_replay_iter)
                    do_replay_num = [1 for _ in range(num_replay_iter)]
                else:
                    do_replay_ids = range(num_training_iter)
                    do_replay_num = [1 for _ in range(num_replay_iter)]
                    remaining_num = num_replay_iter - num_training_iter
                    i = 0
                    while remaining_num != 0:
                        i = (i + 1) % num_training_iter
                        do_replay_num[i] += 1
                        remaining_num -= 1

                replay_generator =  il_trainer.dataloader_replay.__iter__()
                replay_iter_num = 0

                print('Num Replay images: {}'.format(len(il_trainer.dataset_replay)))
                print('Iteration_num: ',len(il_trainer.dataloader_replay))
            # Training Dataset
            for iter_num, data in enumerate(il_trainer.dataloader_train):
                il_trainer.optimizer.param_groups[0]['betas'] = (0.9, 0.999)
                start = time.time()
                if il_trainer.params['agem']:
                    il_trainer.agem.cal_replay_grad(il_loss)
                    
                
                losses = cal_losses(il_trainer, il_loss, data, is_replay=False)
                if losses != None:
                    #continue
                    end = time.time()
                    print_iteration_info(il_trainer, losses, cur_epoch, iter_num, end - start)

                    # Iteration Log
                    epoch_loss.append(losses['total_loss'])
                    avg_times.append(end - start)
            
                    recorder.add_iter_loss(losses)

                #Replay dataset
                if replay_exist and not_warm_classifier and iter_num in do_replay_ids:
                    for i in range(do_replay_num[replay_iter_num]):
                        il_trainer.optimizer.param_groups[0]['betas'] = (il_trainer.params['beta_on_replay'], 0.999)
                        data = replay_generator.next()
                        start = time.time()
                        losses = cal_losses(il_trainer, il_loss, data, is_replay=True)
                        if losses == None:
                            continue
                            
                        end = time.time()
                        print_iteration_info(il_trainer, losses, cur_epoch, replay_iter_num + i, end - start)
                        # Iteration Log
                        epoch_loss.append(losses['total_loss'])
                        avg_times.append(end - start)
                
                        recorder.add_iter_loss(losses)
                    replay_iter_num += 1


            # Replay Dataset
            # if il_trainer.params['agem'] == False and il_trainer.dataset_replay != None and not_warm_classifier:
            #     print("Start Replay!")
            #     print('Num Replay images: {}'.format(len(il_trainer.dataset_replay)))
            #     print('Iteration_num: ',len(il_trainer.dataloader_replay))

            #     il_trainer.optimizer.param_groups[0]['betas'] = (il_trainer.params['beta_on_replay'], 0.999)
            #     for iter_num, data in enumerate(il_trainer.dataloader_replay):
            #         start = time.time()
            #         losses = cal_losses(il_trainer, il_loss, data, is_replay=True)
            #         if losses == None:
            #             continue
            #         end = time.time()
            #         print_iteration_info(il_trainer, losses, cur_epoch, iter_num, end - start)

            #     # Iteration Log
            #     epoch_loss.append(losses['total_loss'])
            #     avg_times.append(end - start)
            #     recorder.add_iter_loss(losses)

            if il_trainer.params['bic'] and il_trainer.bic != None:
                print("Start Bic!")
                il_trainer.bic.bic_training()

            il_trainer.scheduler.step()
            #il_trainer.scheduler.step(np.mean(epoch_loss))
            il_trainer.save_ckp(epoch_loss, epoch=cur_epoch)

            if cur_epoch % 5 == 0:
                il_trainer.auto_delete(cur_state, cur_epoch)


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
        
        

import os
import torch
import torch.nn as nn
from tqdm import tqdm
import time
from os.path import join
from collections import OrderedDict

from utils.tools import load_data,count_parameters,make_optimizer,Logger,AverageMeter
from utils.LossIndex import com_metrics
from utils.lossf import Loss
from utils.visual import *

from models.nn0821 import ours_nn_uxnet_0821


def data_to(inputs,labels):
    device = torch.device('cuda')
    return inputs.float().to(device), labels.float().to(device)


def train_epoch(data_loader, net, optimizer, lossf):
    net.train()
    torch.cuda.empty_cache()
    loss = AverageMeter()
    losses = {loss_name: 0 for loss_name in lossf.loss_names}
    metrics = {name: AverageMeter() for name in ["Precision", "Recall","F1-score", "IoU", "mIoU","HD","HD95"]}

    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    for step, (inputs,labels,filenames) in tqdm(enumerate(data_loader), total=len(data_loader)):
        inputs, labels = data_to(inputs,labels)
        optimizer.zero_grad()
        outputs,lowout = net(inputs)
        step_loss, step_losses = lossf(outputs, labels)
        step_metrics = com_metrics(outputs, labels)
        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        loss.update(step_loss.item(), inputs.size(0))
        for loss_type, loss_value in step_losses.items():
            losses[loss_type] += loss_value 
        for name, value in step_metrics.items():
            metrics[name].update(value, inputs.size(0))

    log = OrderedDict({'train_loss': loss.avg,'lr': current_lr})
    log.update({f'loss_{name.lower()}': value / len(data_loader) for name, value in losses.items()})  # 更新损失项
    log.update({name.lower(): meter.avg for name, meter in metrics.items()}) 
   
    return log, loss.avg


def val_epoch(data_loader,net,lossf):
    net.eval()
    loss = AverageMeter()
    losses = {loss_name: 0 for loss_name in lossf.loss_names}
    metrics = {name: AverageMeter() for name in ["Precision", "Recall", "F1-score",
                                                  "IoU", "mIoU","HD","HD95"]}
    with torch.no_grad():
        for step, (inputs,labels,filenames) in tqdm(enumerate(data_loader), total=len(data_loader)):
            inputs, labels = data_to(inputs,labels)
            outputs,lowout = net(inputs)
            step_loss, step_losses = lossf(outputs, labels)
            step_metrics = com_metrics(outputs, labels)
            loss.update(step_loss.item(), inputs.size(0))
            for loss_type, loss_value in step_losses.items():
                losses[loss_type] += loss_value
            for name, value in step_metrics.items():
                metrics[name].update(value, inputs.size(0))
        log = OrderedDict({'val_loss': loss.avg})
        log.update({f'loss_{name.lower()}': value / len(data_loader) for name, value in losses.items()})  # 更新损失项
        log.update({name.lower(): meter.avg for name, meter in metrics.items()}) 
        
    return log, loss.avg


def train(args):
    # Load data
    print("---")
    print("Loading data ... ")
    train_loader, val_loader = load_data(args)
    print("Load loss fuction")
    lossf = Loss(args)
    print('Create model...')
    ################################################################  对比实验
    # elif args.exp == '0501resunet':
    #     model = ResUNet(n_channels=1, n_classes=1)

    if args.exp == 'nn_unxet_0821':
        model = ours_nn_uxnet_0821()

        

    model = model.to(torch.device('cpu' if args.cpu else 'cuda'))
    if args.n_GPUs > 1:
        model = nn.DataParallel(model, range(args.n_GPUs))
    print("Total number of parameters: " + str(count_parameters(model)))

    if args.load_pred_save_model == 'True':
        checkpoint = torch.load(args.pretrained_model_name, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = checkpoint['net']
        if any(k.startswith("module.") for k in state_dict.keys()):
            print("Detected DataParallel model, removing 'module.' prefix...")
            state_dict = {k.replace("module.", "",1): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict) 

    model_path = './EXP/' + args.exp  +'/models/'
    print("The model is saved in : ", model_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    print("Define optimizer ... ")
    optimizer = make_optimizer(args, model)
    # start training
    print("---")
    print("Start training ... ")

    save_path = join('./EXP/', args.exp) #  , args.attr_mode)
    log_train = Logger(save_path,'trainlog')
    log_val = Logger(save_path, 'vallog')

    best_val_iou = 0.0
    best_val_loss = 1000.0   

    for epoch in range(1,args.epochs+1):
        print('\nEPOCH: %d/%d --(learn_rate:%.6f) | Time: %s' % \
              (epoch, args.epochs, optimizer.state_dict()['param_groups'][0]['lr'], time.asctime()))
        

        train_log, train_loss = train_epoch(train_loader, model, optimizer,lossf)
        log_train.update(epoch, train_log)

        val_log, val_loss= val_epoch(val_loader, model,lossf)
        log_val.update(epoch, val_log)

        if args.scheduler =='MultiStepLR':
             optimizer.schedule()
        elif args.scheduler =='ReduceLROnPlateau':
            optimizer.schedule(val_loss)

        if val_loss  < best_val_loss:
            best_val_loss = val_loss 
            best_val_loss_epoch = epoch
            state_val_best_loss = {'net': model.state_dict(), 
                                   'optimizer': optimizer.state_dict(), 'epoch': best_val_loss_epoch}

        if val_log["iou"]  > best_val_iou:
            best_val_iou, best_val_iou_epoch = val_log["iou"], epoch
            state_val_best_iou = {'net': model.state_dict(), 
                                  'optimizer': optimizer.state_dict(), 'epoch': best_val_iou_epoch}


        if epoch % args.save_model_every == 0:
            state_val_every = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
            model_name = 'epoch_{}_tiou_{:.4f}_viou_{:.4f}_CP.pth'.format(epoch,
                                                train_log["iou"],val_log["iou"])  
            torch.save(state_val_every, model_path + model_name)
            
            
        if epoch % args.random_visual == 0:
            visual_result_3d(args,model,epoch,'train')
            visual_result_3d(args,model,epoch,'valid')
            print('save train/valid result')

        if epoch % args.field_visual == 0:
            field_result_3d(args,model,epoch)
            print('save f3, kerry, shengli _dim0/1/2 and 3d result')
            
    torch.cuda.empty_cache()

    torch.save(state_val_best_loss, join(model_path,f'val_best_loss_model_{best_val_loss_epoch}.pth'))
    print(" best val loss ({:.6f} --> epoch {:.6f}). ".format(best_val_loss, best_val_loss_epoch))
    torch.save(state_val_best_iou, join(model_path,f'val_best_iou_model_{best_val_iou_epoch}.pth'))
    print(" best val iou ({:.6f} --> epoch {:.6f}). ".format(best_val_iou, best_val_iou_epoch))


    print("---")
    print("Train Finish !")

    return 0


import os
import torch
from dataloader.dataloader import FaultDataset3d
from torch.utils.data import DataLoader,ConcatDataset


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2


import torch.optim as optim  # 优化器模块(SGD,Adam,RMSprop)
import torch.optim.lr_scheduler as lrs  

import time
from collections import OrderedDict



def pad_field_to_128(field):
    original_shape = field.shape
    pad_depth = 128 - original_shape[0]
    padded = np.pad(field, ((0, pad_depth), (0, 0), (0, 0)), mode='constant', constant_values=0)
    return padded, original_shape[0]  # 返回填充后数据，以及原始 depth 大小

def recover_original_field(padded_field, original_depth):
    return padded_field[:original_depth]


def save_args_info(args):
    # save args to config.txt
    argsDict = args.__dict__
    result_path = './EXP/' + '/' + args.exp + '/' # + args.attr_mode + '/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if args.mode == 'train':
        with open(result_path + 'config_train.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    elif args.mode == 'valid':
        with open(result_path + 'config_valid.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    elif args.mode == 'pred':
        with open(result_path + 'config_pred.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data(args):
    # args.mode=['train', 'valid'] 
    if args.mode == 'train':
        train_dataset = FaultDataset3d(args.train_path, args.data_auge,kind = 'train')
        valid_dataset = FaultDataset3d(args.valid_path, args.data_auge,kind = 'valid')
        if args.data_val_add =='True':
            train_dataset = ConcatDataset([train_dataset, valid_dataset])

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                      pin_memory=not args.cpu, num_workers=args.workers, drop_last=True)
        
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size_not_train, shuffle=False,
                                      pin_memory=not args.cpu, num_workers=args.workers, drop_last=True)
        
        print("--- create train dataloader ---")
        print(len(train_dataset), ", train dataset created")
        print(len(train_dataloader), ", train dataloader created")
        print("--- create valid dataloader ---")
        print(len(valid_dataset), ", valid dataset created")
        print(len(valid_dataloader), ", valid dataloaders created")
        return train_dataloader, valid_dataloader

    elif args.mode == 'valid':
        valid_dataset = FaultDataset3d(args.valid_path, args.data_auge,kind = 'valid')
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size_not_train, shuffle=False, 
                                      num_workers=args.workers, drop_last=True)
        print("--- create valid dataloader ---")
        print(len(valid_dataset), ", valid dataset created")
        print(len(valid_dataloader), ", valid dataloaders created")
        return valid_dataloader


def make_optimizer(args, target):
    ''' 
    Make optimizer and scheduler without warm-up
    target: model 
    '''
    # optimizer
    if args.optimizer == 'ADAMW':
        # 添加对 AdamW 的支持
        decay_params = []
        no_decay_params = []
        for name, param in target.named_parameters():
            if not param.requires_grad:
                continue
            if 'bias' in name or 'bn' in name or 'norm' in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        param_groups = [
            {'params': decay_params, 'weight_decay': args.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
    else:
        param_groups = filter(lambda x: x.requires_grad, target.parameters())
    
    kwargs_optimizer = {'lr': args.lr}
    
    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer.update({
            'momentum': args.momentum,
            'weight_decay': args.weight_decay
        })
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer.update({
            'betas': (0.9, 0.999),
            'eps': args.epsilon,
            'weight_decay': args.weight_decay
        })
    elif args.optimizer == 'ADAMW':
        optimizer_class = optim.AdamW
        kwargs_optimizer.update({
            'betas': (0.9, 0.999),
            'eps': args.epsilon
            # weight_decay 已在 param_groups 中设置
        })
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer.update({
            'eps': args.epsilon,
            'weight_decay': args.weight_decay
        })

    # 选择调度器
    if args.scheduler == 'MultiStepLR':
        milestones = list(map(lambda x: int(x), args.decay.split('-')))
        kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
        scheduler_class = lrs.MultiStepLR
    elif args.scheduler == 'ReduceLROnPlateau':
        kwargs_scheduler = {'mode': 'min', 'factor': 0.9, 'patience': 5, 'verbose': True}
        scheduler_class = lrs.ReduceLROnPlateau
    elif args.scheduler == 'CosineAnnealingLR':
        kwargs_scheduler = {'T_max': args.epochs, 'eta_min': args.min_lr}
        scheduler_class = lrs.CosineAnnealingLR

    class CustomOptimizer(optimizer_class):
        # 继承PyTorch中的优化器 optimizer_class 的所有方法和属性
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)
            self.base_lr = kwargs.get('lr', 0.1)  # 基础学习率
            self.current_epoch = 0  # 当前的epoch
        
        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)
        
        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))
        
        def load(self, load_dir, epoch=1):
            self.load_state_dict(torch.load(self.get_dir(load_dir)))
            if epoch > 1:
                for _ in range(epoch):
                    self.scheduler.step()
        
        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')
        
        def schedule(self):
            # 使用普通scheduler更新学习率
            self.scheduler.step()

        def get_lr(self):
            # 返回调度器的当前学习率
            return self.scheduler.get_last_lr()[0]
        
        def get_last_epoch(self):
            return self.scheduler.last_epoch

    optimizer = CustomOptimizer(param_groups, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    
    return optimizer


def dict_round(dic,num):
    for key,value in dic.items():
        dic[key] = round(value,num)
    return dic


class Logger():
    def __init__(self,save_name,name):
        self.log = None
        self.summary = None
        self.save = save_name
        self.name = name
        self.time_now = time.strftime('_%Y-%m-%d-%H-%M', time.localtime())

    def update(self,epoch,train_log):
        item = OrderedDict({'epoch':epoch})
        item.update(train_log)
        item = dict_round(item,4) # 保留小数点后6位有效数字
        print(item)
        self.update_csv(item)
        # self.update_tensorboard(item)

    def update_csv(self,item):
        tmp = pd.DataFrame(item,index=[0])
        if self.log is not None:
            # self.log = self.log.append(tmp, ignore_index=True)
            self.log = pd.concat([self.log, tmp])
        else:
            self.log = tmp
        self.log.to_csv('%s/%s %s.csv' %(self.save,self.name,self.time_now), index=False)


class AverageMeter(object):
    """Computes and stores the average and current value for calculate average loss"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



##
'save-result-valid 没改'
def save_result(args, segs, inputs, gts, val_loss, val_iou, val_dice):
    result_path = './EXP/' + args.exp + '/results/valid/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + "valid_final_result.txt", 'a+') as f:
        f.write('valid loss:\t' + str(val_loss) + '\n')
        f.write('valid iou:\t' + str(val_iou) + '\n')
        f.write('valid dice:\t' + str(val_dice) + '\n')

    # if not os.path.exists(result_path + '/numpy/'):
    #     os.makedirs(result_path + '/numpy/')
    if not os.path.exists(result_path + '/picture/'):
        os.makedirs(result_path + '/picture/')

    for i in range(len(inputs)):

        seg = segs[i].argmax(axis=1)
        img = inputs[i]
        gt = gts[i]
        seg = np.squeeze(seg)
        img = np.squeeze(img)
        gt = np.squeeze(gt)
        # # save output
        # np.save(result_path + '/numpy/' + str(i) + '_seg.npy', seg)
        # np.save(result_path + '/numpy/' + str(i) + '_img.npy', img)
        # np.save(result_path + '/numpy/' + str(i) + '_gt.npy', gt)
        # save picture

        index = np.arange(0, 128, 50)
        for idx in index:
            # dim 0
            plt.subplot(1, 3, 1)
            plt.imshow(np.transpose(img[idx, :, :]),cmap='gray')
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(np.transpose(gt[idx, :, :]),cmap='gray')
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(np.transpose(seg[idx, :, :]),cmap='gray')
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_0.png')
            plt.close()
            # dim 1
            plt.subplot(1, 3, 1)
            plt.imshow(np.transpose(img[:, idx, :]),cmap='gray')
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(np.transpose(gt[:, idx, :]),cmap='gray')
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(np.transpose(seg[:, idx, :]),cmap='gray')
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_1.png')
            plt.close()
            # dim 2
            plt.subplot(1, 3, 1)
            plt.imshow(np.transpose(img[:, :, idx]),cmap='gray')
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(np.transpose(gt[:, :, idx]),cmap='gray')
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(np.transpose(seg[:, :, idx]),cmap='gray')
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_2.png')
            plt.close()


# #####################################################
# def show_result_v0(input,output,save_path,num,dim,pos):
#     input_normalized= cv2.normalize(input, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#     height, width = output.shape
#     color_map = np.zeros((height, width, 3), dtype=np.uint8)
#     for i in range(height):
#         for j in range(width):
#             prob = output[i, j]
#             if prob < 0.4:
#                 color_map[i, j] = [0, 0, 0]  # 设置为黑色（透明）
#             else:
#                 # 将 0.4 到 1 之间的值线性映射到 [0, 255]
#                 normalized_value = int((prob - 0.4) / 0.6 * 255)
#                 color_map[i, j] = cv2.applyColorMap(np.array([[normalized_value]], dtype=np.uint8), cv2.COLORMAP_JET)[0][0]

#     input_colored = cv2.cvtColor(input_normalized, cv2.COLOR_GRAY2BGR)
#     overlay = cv2.addWeighted(input_colored,  1, color_map, 0.5, 0)
#     cv2.imwrite(save_path + str(num)+'_dim'+str(dim) +'_'+str(pos)+'.png',overlay)



'v0'
# def field_result_plt(args,net,num):         
#     net.eval()
#     with torch.no_grad():
        
#         data = np.fromfile('./dataset/f3d/gxl.dat', dtype=np.single)
#         data = data - np.mean(data)
#         data = data / np.std(data)
#         input_data = np.reshape(data,(512,384,128))
        
        
#         pred_path = './EXP/' + args.exp + '/results'+'/f3/' 
#         if not os.path.exists(pred_path):
#             os.makedirs(pred_path)
            
#         overlap = args.overlap 
           
#         input_2 = input_data[:,:,80]
#         output_data_2 = sliding_window_prediction(input_2, net, overlap,block_size=128)         
#         plt.subplot(1, 2, 1)
#         plt.imshow(input_2, cmap='gray')
#         plt.subplot(1, 2, 2)
#         plt.imshow(output_data_2, cmap='gray')
#         plt.savefig(pred_path +str(num)+ '_'+'80_dim2.png', dpi=600)
#         plt.close()
#         input_1 = input_data[:,80,:]
#         output_data_1 = sliding_window_prediction(input_1, net, overlap,block_size=128)         
#         plt.subplot(1, 2, 1)
#         plt.imshow(input_1, cmap='gray')
#         plt.subplot(1, 2, 2)
#         plt.imshow(output_data_1, cmap='gray')
#         plt.savefig(pred_path +str(num)+ '_'+'80_dim1.png', dpi=600)
#         plt.close()
#         input_0 = input_data[80,:,:]
#         output_data_0 = sliding_window_prediction(input_0, net, overlap,block_size=128)         
#         plt.subplot(1, 2, 1)
#         plt.imshow(input_0, cmap='gray')
#         plt.subplot(1, 2, 2)
#         plt.imshow(output_data_0, cmap='gray')
#         plt.savefig(pred_path +str(num)+ '_'+'80_dim0.png', dpi=600)
#         plt.close()



# def ours_result_mask_plt(args,net,num):         
#     net.eval()
#     with torch.no_grad():
        
#         # data = np.fromfile('./dataset/ours/data_fi.npy', dtype=np.single)
#         data = np.load('./dataset/ours/data_fi.npy')
#         data = data - np.mean(data)
#         data = data / np.std(data)
        
#         pred_path = './EXP/' + args.exp + '/results'+'/ours/' 
#         if not os.path.exists(pred_path):
#             os.makedirs(pred_path)
#         overlap = args.overlap
        
#         input_0 = data[20,:,:]
#         output_data_0 = sliding_window_prediction(input_0, net, overlap,block_size=128)
#         input_0 = np.transpose(input_0)
#         output_data_0 = np.transpose(output_data_0)
#         plt.subplot(1, 2, 1)
#         plt.imshow(input_0, cmap='gray')
#         plt.subplot(1, 2, 2)
#         plt.imshow(input_0, cmap='gray')  # 显示原始图像
#         red_mask = np.zeros((output_data_0.shape[0], output_data_0.shape[1], 3), dtype=np.uint8)
#         red_mask[output_data_0 == 1] = [255, 0, 0]    
#         plt.imshow(red_mask, alpha=0.5)
#         plt.savefig(pred_path +str(num)+ '_'+'20_dim0.png', dpi=600)
#         plt.close()
        
#         input_1 = data[:,20,:]
#         output_data_1 = sliding_window_prediction(input_1, net, overlap,block_size=128)
#         input_1 = np.transpose(input_1)
#         output_data_1 = np.transpose(output_data_1)
#         plt.subplot(1, 2, 1)
#         plt.imshow(input_1, cmap='gray')
#         plt.subplot(1, 2, 2)
#         plt.imshow(input_1, cmap='gray')  # 显示原始图像
#         red_mask = np.zeros((output_data_1.shape[0], output_data_1.shape[1], 3), dtype=np.uint8)
#         red_mask[output_data_1 == 1] = [255, 0, 0]    
#         plt.imshow(red_mask, alpha=0.5)
#         plt.savefig(pred_path +str(num)+ '_'+'20_dim1.png', dpi=600)
#         plt.close()
        
#         input_2 = data[:,:,20]
#         output_data_2 = sliding_window_prediction(input_2, net, overlap,block_size=128)
#         input_2 = np.transpose(input_2)
#         output_data_2 = np.transpose(output_data_2)
#         plt.subplot(1, 2, 1)
#         plt.imshow(input_2, cmap='gray')
#         plt.subplot(1, 2, 2)
#         plt.imshow(input_2, cmap='gray')  # 显示原始图像
#         red_mask = np.zeros((output_data_2.shape[0], output_data_2.shape[1], 3), dtype=np.uint8)
#         red_mask[output_data_2 == 1] = [255, 0, 0]    
#         plt.imshow(red_mask, alpha=0.5)
#         plt.savefig(pred_path +str(num)+ '_'+'20_dim2.png', dpi=600)
#         plt.close()
        
 
 

    






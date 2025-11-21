import os
import torch
from dataloader.dataloader import FaultDataset3d
from torch.utils.data import DataLoader
import torch.optim as optim  
import torch.optim.lr_scheduler as lrs  
import time
from collections import OrderedDict


def save_args_info(args):
    argsDict = args.__dict__
    result_path = './EXP/' + '/' + args.exp + '/' 
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if args.mode == 'train':
        with open(result_path + 'config_train.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    elif args.mode == 'test':
        with open(result_path + 'config_test.txt', 'w') as f:
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

    elif args.mode == 'test':
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
        })
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer.update({
            'eps': args.epsilon,
            'weight_decay': args.weight_decay
        })


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
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)
            self.base_lr = kwargs.get('lr', 0.1)  
            self.current_epoch = 0  
        
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
            self.scheduler.step()

        def get_lr(self):
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
        item = dict_round(item,4) 
        print(item)
    #     self.update_csv(item)

    # def update_csv(self,item):
    #     tmp = pd.DataFrame(item,index=[0])
    #     if self.log is not None:
    #         self.log = pd.concat([self.log, tmp])
    #     else:
    #         self.log = tmp
    #     self.log.to_csv('%s/%s %s.csv' %(self.save,self.name,self.time_now), index=False)


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


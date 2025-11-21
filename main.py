# Fault Segmentation Based on Pytorch
import os
import argparse
import torch
import warnings
import random
import numpy as np
from utils.train import train
from utils.tools import save_args_info
from utils.test import *

def add_args():
    parser = argparse.ArgumentParser(description="3d_FaultSeg_pytorch_edge_region")
    
    parser.add_argument("--exp", default="ours", type=str, help="Name of each run")  
    parser.add_argument("--mode", default='train', choices=['train','test','pred'], type=str)

    parser.add_argument('--loss', default="0.5*dice+0.5*bce")
    parser.add_argument("--epochs", default=1, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size") 
    
    parser.add_argument('--lr', type=float, default=0.00015,help='learning rate') 
    parser.add_argument('--decay', type=str, default='10-20-25-30-35-40-45-50', help='learning rate decay type,scheduler')
    parser.add_argument('--optimizer', default='ADAMW',choices=('SGD', 'ADAM','ADAMW', 'RMSprop'))
    parser.add_argument('--weight_decay', default=1e-4,help='weight decay')   
    parser.add_argument('--momentum' ,default= '0.9',type=float, help='SGD momentum') 
    parser.add_argument('--epsilon',type=float,default= '1e-8', help='ADAM epsilon for numerical stability')
    parser.add_argument('--scheduler',type=str,default= 'MultiStepLR')  
    parser.add_argument('--gamma', type=float, default=0.9,help='learning rate decay factor for step decay')
       
    parser.add_argument('--overlap', default=0.5, type=float, help='preds overlap')
    parser.add_argument('--threshold', default=0.5, type=float, help='Classification threshold')
    parser.add_argument('--sigma', default=0.0, type=float, help='Gaussian filter sigma')
    parser.add_argument("--batch_size_not_train", default=1, type=int, help="number of batch size when not training")
    parser.add_argument('--cpu', action='store_true',help='use cpu only')
    parser.add_argument("--device", default='cuda', type=str, help="GPU id for training")
    parser.add_argument('--n_GPUs', type=int, default=2, help='number of GPUs')
    parser.add_argument("--workers", default=10, type=int, help="number of workers")

    parser.add_argument("--train_path", default="", type=str, help="dataset directory")
    parser.add_argument("--valid_path", default="", type=str, help="dataset directory")
    parser.add_argument("--data_auge", default="True", type=str, help="data augmentation True/False")

    parser.add_argument("--pretrained_model_path", 
                        default="", 
                        type=str, help="pretrained model path and name")

    parser.add_argument("--pre_data",default='',type=str)  
    parser.add_argument("--pre_path",default='',type=str)

    args = parser.parse_args()

    print()
    print(">>>============= args ====================<<<")
    print()
    print(args)  
    print()
    print(">>>=======================================<<<")

    return args

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args):
    set_seed(0)  # 3407
    save_args_info(args)
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode =='pred':
        pred(args)


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    args = add_args()
    main(args)

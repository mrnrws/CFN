# Fault Segmentation Based on Pytorch
import os
import argparse
import torch
import warnings
import random
import numpy as np

from utils.train import train
# from utils.test import field,valid


from utils.tools import save_args_info


def add_args():
    parser = argparse.ArgumentParser(description="3d_FaultSeg_pytorch_edge_region")

    
    parser.add_argument("--exp", default="nn_unxet_0821", type=str, help="Name of each run")  # gai114_HE2Ddsnet_wt
    # nn_uxnet,nn_mix,nn_mix_0821,nn_unxet_0821,nn_unxet_0825
    # nn_unxet_0822
    # nn_MIX_HED
    parser.add_argument("--mode", default='train', choices=['train', 'valid', 'pred'], type=str)

    parser.add_argument('--loss', default="0.5*dice+0.5*bce",
                          help='loss function configuration, loss*MSSSIM+(1-loss)*L1')
    # str(0.5)+'*dice+'+str(0.5)+'*bce' # str(0.5)+'*Bbce+'+ str(0.5)+'*maskdice'
    # 0.3*dice+0.3*bce+0.4*msssim,  0.5*dice+0.2*Bbce+0.3*focal

    parser.add_argument("--epochs", default=200, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")

    parser.add_argument("--save_model_every", default=10, type=int, help="save model every n epochs")
    parser.add_argument('--random_visual', default=10, type=int, help='visual  syn result')
    parser.add_argument('--field_visual', default=10, type=int, help='visual f3 result')  
    
    parser.add_argument('--lr', type=float, default=0.00015,help='learning rate') # 0.0001 1e-4  # 0.001
    # 3dfaultseg 0.00015  10-20-25-30-35-40-45-50
    # vsmix 0.0005
    parser.add_argument('--decay', type=str, default='10-20-25-30-35-40-45-50', help='learning rate decay type,scheduler')
    # 10-20-30-40-50-60-65-70-75-80-85-90-95  
    parser.add_argument('--optimizer', default='ADAMW',choices=('SGD', 'ADAM','ADAMW', 'RMSprop'))
    parser.add_argument('--weight_decay', default=1e-4,help='weight decay')   # ResNet 1e-4 ~ 1e-2; ViT 1e-2  减少过拟合
    parser.add_argument('--momentum' ,default= '0.9',type=float, help='SGD momentum') 
    parser.add_argument('--epsilon',type=float,default= '1e-8', help='ADAM epsilon for numerical stability')
    parser.add_argument('--scheduler',type=str,default= 'MultiStepLR')  # MultiStepLR, ReduceLROnPlateau
    parser.add_argument('--gamma', type=float, default=0.9,help='learning rate decay factor for step decay')
       
    parser.add_argument('--overlap', default=0.5, type=float, help='preds overlap')
    parser.add_argument('--threshold', default=0.5, type=float, help='Classification threshold')
    parser.add_argument('--sigma', default=0.0, type=float, help='Gaussian filter sigma')
    parser.add_argument("--batch_size_not_train", default=1, type=int, help="number of batch size when not training")
    parser.add_argument('--cpu', action='store_true',help='use cpu only')
    parser.add_argument("--device", default='cuda', type=str, help="GPU id for training")
    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
    parser.add_argument("--workers", default=10, type=int, help="number of workers")

    parser.add_argument("--train_path", default="/home/user/data/zwt/datasets/FaultDataset/train", type=str, help="dataset directory")
    parser.add_argument("--valid_path", default="/home/user/data/zwt/datasets/FaultDataset/valid", type=str, help="dataset directory")
    parser.add_argument("--data_val_add", default="False", type=str)  # False True
    parser.add_argument("--data_auge", default="True", type=str, help="data augmentation True/False")

    parser.add_argument("--load_pred_save_model", default="False", type=str)  # False True
    parser.add_argument("--pretrained_model_name", 
                        default="/root/data/vsr/0427/EXP/ours_05050428_v0/None/models/epoch_45_tiou_0.6844_viou_0.7008_CP.pth", 
                        type=str, help="pretrained model path and name")
    
    parser.add_argument("--pred_data_name", default="ours", choices=['f3', 'kerry','ours'], type=str, help="pretrained data name")
    parser.add_argument("--pred_pos", default="172",  type=int, help="inline 0,time 1")
    parser.add_argument("--pred_dim", default="1", type=int)

    args = parser.parse_args()

    print()
    print(">>>============= args ====================<<<")
    print()
    print(args)  # print command line args
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
    # elif args.mode == 'field':
    #     field(args)
    # elif args.mode == 'valid':
    #     valid(args)

    else:
        raise ValueError("Only ['train', 'field','valid'] mode is supported.")
    


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" # 0,1
    args = add_args()
    main(args)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
from importlib import import_module
import torch
import torch.nn as nn
import torch.nn.parallel as P
from models.faultseg3d import FaultSeg3D

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        print('Making model...')
        print(torch.cuda.device_count())
        self.device = torch.device('cuda')
        self.n_GPUs = args.n_GPUs

        self.model = FaultSeg3D(args.in_channels, args.out_channels)

    def forward(self, x):
        if self.n_GPUs > 1:
            return P.data_parallel(self.model, x, range(self.n_GPUs))
        else:
            return self.model(x)
        #else:
            #forward_function = self.model.forward
            #return forward_function(x)
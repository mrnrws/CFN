import torch
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F



class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(2),
                                        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                                        nn.BatchNorm3d(out_channels),nn.ReLU(inplace=True))
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
                                  nn.BatchNorm3d(out_channels),nn.ReLU(inplace=True))
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


######################################
class subnet1(nn.Module):
    def __init__(self, in_channels):
        super(subnet1, self).__init__()

        self.ec1 = nn.Sequential(nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(32),nn.ReLU(inplace=True))
        self.ec2 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(32),nn.ReLU(inplace=True))
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = nn.Sequential(nn.MaxPool3d(2),
                                     nn.Conv3d(128, 512, kernel_size=3, stride=1, padding=2,dilation=2),
                                     nn.BatchNorm3d(512),nn.ReLU(inplace=True))

        self.up3 = Up(640, 128)
        self.up2 = Up(192, 64)
        self.up1 = Up(96, 32)

    def forward(self, x):
        x1 = self.ec1(x)     
        x2 = self.ec2(x1)    
        e1 = self.down1(x2)  
        e2 = self.down2(e1)  
        e3 = self.down3(e2) 

        d2 = self.up3(e3, e2)  
        d1 = self.up2(d2, e1)  
        d0 = self.up1(d1, x2)  
        out = d0+x1

        return out 


##################################
class subnet2(nn.Module):
    def __init__(self, in_channels):
        super(subnet2, self).__init__()

        self.ec1 = nn.Sequential(nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(64),nn.ReLU(inplace=True))
        self.ec2 = nn.Sequential(nn.Conv3d(64, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(64),nn.ReLU(inplace=True))
        self.down1 = Down(64, 128)
        self.down2 = nn.Sequential(nn.MaxPool3d(2),
                                     nn.Conv3d(128, 512, kernel_size=3, stride=1, padding=2,dilation=2),
                                     nn.BatchNorm3d(512),nn.ReLU(inplace=True))

        self.up2 = Up(640, 128)
        self.up1 = Up(192, 64)

    def forward(self, x):
        x1 = self.ec1(x)     
        x2 = self.ec2(x1)    
        e1 = self.down1(x2)  
        e2 = self.down2(e1)  

        d1 = self.up2(e2, e1)  
        d0 = self.up1(d1, x2)  
        return d0 + x1  

##################################
class subnet3(nn.Module):
    def __init__(self, in_channels):
        super(subnet3, self).__init__()

        self.ec1 = nn.Sequential(nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(16),nn.ReLU(inplace=True))
        self.ec2 = nn.Sequential(nn.Conv3d(16, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(16),nn.ReLU(inplace=True))

        self.dc1 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=2,dilation=2),
                                   nn.BatchNorm3d(32),nn.ReLU(inplace=True))
        self.dc2 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=4,dilation=4),
                                   nn.BatchNorm3d(32),nn.ReLU(inplace=True))
        self.dc3 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=8,dilation=8),
                                   nn.BatchNorm3d(64),nn.ReLU(inplace=True))

        self.dc3 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=8,dilation=8),
                                   nn.BatchNorm3d(64),nn.ReLU(inplace=True))
        self.dc3 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=8,dilation=8),
                                   nn.BatchNorm3d(64),nn.ReLU(inplace=True))  

        self.dc22 = nn.Sequential(nn.Conv3d(96, 32, kernel_size=3, stride=1, padding=4,dilation=4),
                                   nn.BatchNorm3d(32),nn.ReLU(inplace=True))                                 
        self.dc21 = nn.Sequential(nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=4,dilation=4),
                                   nn.BatchNorm3d(32),nn.ReLU(inplace=True)) 
        self.dec0 = nn.Sequential(nn.Conv3d(48, 16, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(16),nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.ec1(x)     
        x2 = self.ec2(x1)    
        e1 = self.dc1(x2)     
        e2 = self.dc2(e1)     
        e3 = self.dc3(e2)     
        # 
        d2 = self.dc22(torch.cat([e3,e2], dim=1)) 
        d1 = self.dc21(torch.cat([d2,e1], dim=1)) 
        d0 = self.dec0(torch.cat([d1,x2], dim=1)) 

        return d0+x1  


###################################
class edgeup(nn.Module):
    def __init__(self, in_channels, out_channels,up_sacle):  
        super(edgeup, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1) # , stride=1, padding=1)
        self.upsample_volume = nn.Upsample(scale_factor=(up_sacle, up_sacle, up_sacle), mode='trilinear', align_corners=True)

    def forward(self, x):
        x = nn.Sigmoid()(self.conv(x))
        out = self.upsample_volume(x)
        return out


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

##################################################################
class nrunet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(nrunet, self).__init__()
        self.n_channels = n_channels

        self.down1 = subnet1(n_channels)  
        self.down2 = subnet2(32)  
        self.down3 = subnet3(64)  

        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)

        self.dec2 = subnet2(80)  
        self.dec1 = subnet1(96) 

        self.edgeup3 = edgeup(16,1,4)
        self.edgeup2 = edgeup(64,1,2)
        self.edge1 = OutConv(32,1)

        self.outc = OutConv(16, n_classes)
        self.fioutc = OutConv(3, n_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x1 = nn.MaxPool3d(2)(self.down1(x))  
        x2 = nn.MaxPool3d(2)(self.down2(x1)) 
        x3 = self.down3(x2)                 
        

        d2 = self.dec2(self.up(torch.cat([x3,x2], dim=1)))  
        d1 = self.dec1(self.up(torch.cat([d2,x1], dim=1)))  

        ed3 = self.edgeup3(x3)
        ed2 = self.edgeup2(d2)
        ed1 = nn.Sigmoid()(self.edge1(d1))

        out = self.fioutc(torch.cat([ed3,ed2,ed1], dim=1))
        outputs = self.sigmoid(out)
        return outputs
    


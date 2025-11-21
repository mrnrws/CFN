import torch.nn as nn
import torch
import torch.nn.functional as F

# self.bn2 = nn.InstanceNorm3d(out_channels)
# self.relu = nn.LeakyReLU(inplace=True)

'DoubleConv' 'ux_block'
'SCA3D'
'DecoderBlock--UnetDecoderHead'  # nn.ConvTranspose3d 2*out_c--out_c  
'Down' # nn.Sequential(nn.MaxPool3d(2),DoubleConv(in_channels, out_channels))
# 应该用不到
'PixelResBottleneck'  # 标准瓶颈结构: 1x1 -> 3x3 -> 1x1 卷积，并带有残差连接
'DirectionalSurfaceConvBlock3D--UP_dir'
'EdgeBlock' 'SegSEBlock' 'MCF_Block'

################################################## ConvBlock
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.InstanceNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.InstanceNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

       
######################################################  ux_block
class ux_block(nn.Module):
    def __init__(self, dim): #, drop_path=0.):
        super().__init__()

        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)  # 深度可分离卷积 groups=dim
        # 每个通道独立计算，相当于 3D 深度可分离卷积，减少计算量，同时保留局部特征 # 保留空间信息，降低参数量。
        self.norm = nn.GroupNorm(1,dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim)  # 逐点卷积 ,通道扩展
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim)  # 逐点卷积 
        # self.dropout = nn.Dropout3d(p=0.1)
        # 随机丢弃整个残差分支，增强模型鲁棒性,drop_path=0. 时等于什么都不做
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)  #  LayerNorm
        x = self.pwconv1(x)
        x = self.act(x)
        # x = self.dropout(x)
        x = self.pwconv2(x)
        x = input + x
        return x



#################################'标准瓶颈结构: 1x1 -> 3x3 -> 1x1 卷积，并带有残差连接'
class PixelResBottleneck(nn.Module):
    expansion = 2
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super(PixelResBottleneck, self).__init__()
        self.conv1 = nn.Conv3d(in_c, out_c, kernel_size=1, bias=False)
        self.ln = nn.GroupNorm(1,out_c)
        self.conv2 = nn.Conv3d(out_c, out_c * self.expansion, kernel_size=3, stride=stride, padding=1, bias=False)
        self.ln2 = nn.GroupNorm(1,out_c * self.expansion)
        self.conv3 = nn.Conv3d(out_c * self.expansion, out_c , kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = self.relu(self.ln(self.conv1(x)))
        out = self.relu(self.ln2(self.conv2(out)))
        out = self.ln(self.conv3(out)) + identity
        out = self.relu(out)
        return out


##################################################   SCA3D
class SCA3D(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv3d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False)  # (batch, 1, depth, height, width)

    def forward(self, x):
        bahs, chs, _, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)  
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1,1))
        chn_se = torch.mul(x, chn_se)  # 逐通道相乘
        spa_se = torch.sigmoid(self.spatial_se(x)) # 
        spa_se = torch.mul(x, spa_se)  # 逐元素相乘
        net_out = spa_se + x + chn_se
        return net_out


###############################################  DecoderBlock
class DecoderBlock(nn.Module):           
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels * 2, out_channels)  # *2 for concat with skip connection
        
    def forward(self, x, skip_features):
        x = self.upconv(x)
        if x.shape[2:] != skip_features.shape[2:]:
            x = F.interpolate(x, size=skip_features.shape[2:], mode='trilinear', align_corners=True)    

        x = torch.cat((skip_features, x), dim=1)
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels))
    def forward(self, x):
        return self.maxpool_conv(x)


####################################################  DirectionalSurfaceConvBlock3D
class DirectionalSurfaceConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(DirectionalSurfaceConvBlock3D, self).__init__()

        mid_channels = in_channels // 2  # 稍微加大一点中间通道，提升表达力
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1)
        self.ln1 = nn.GroupNorm(1, mid_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # 四个方向的卷积
        # YZ 面：span{(0,1,0),(0,0,1)} —— 垂直走向 + 深度外挤
        self.conv0 = nn.Conv3d(mid_channels, mid_channels // 4, kernel_size=(1, 9, 9), 
                               padding=(0, 4 * dilation_rate, 4 * dilation_rate), dilation=(1, dilation_rate, dilation_rate))
        # XZ 面：span{(1,0,0),(0,0,1)} —— 水平走向 + 深度外挤                       
        self.conv90 = nn.Conv3d(mid_channels, mid_channels // 4, kernel_size=(9, 1, 9), 
                                padding=(4 * dilation_rate, 0, 4 * dilation_rate), dilation=(dilation_rate, 1, dilation_rate))
        # 45° / 135° 在 inline 切面上的， crossline 外挤                  
        self.conv45 = nn.Conv3d(mid_channels, mid_channels // 4, kernel_size=(9, 9, 1), 
                                padding=(4 * dilation_rate, 4 * dilation_rate, 0), dilation=(dilation_rate, dilation_rate, 1))
        self.conv135 = nn.Conv3d(mid_channels, mid_channels // 4, kernel_size=(9, 9, 1), 
                                 padding=(4 * dilation_rate, 4 * dilation_rate, 0), dilation=(dilation_rate, dilation_rate, 1))

        self.conv_out = nn.Conv3d(mid_channels, out_channels, kernel_size=1)
        self.ln3 = nn.GroupNorm(1, out_channels)

        self._init_weights()

    def forward(self, x):
        x = self.relu(self.ln1(self.conv1(x)))

        x0 = self.conv0(x)
        x90 = self.conv90(x)
        x45 = self.inv_dw_transform_plus45(self.conv45(self.dw_transform_plus45(x)),W=x.size(-1))
        x135 = self.inv_dw_transform_135(self.conv135(self.dw_transform_135(x)), W=x.size(-1))

        x = torch.cat([x0, x90, x45, x135], dim=1)
        x = self.conv_out(x)
        x = self.ln3(x)
        x = self.relu(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def dw_transform_plus45(self, x):
        # x: (B,C,D,H,W) → xs: (B,C,D,H,W'),   W' = W + (D-1)
        B,C,D,H,W = x.shape
        L, R = (D-1), 0
        Wp = W + L + R
        # 只在 W 维左侧补 L 个
        x_pad = F.pad(x, (L, R, 0, 0, 0, 0))   # pad 顺序: (Wl,Wr, Hl,Hr, Dl,Dr)

        base = torch.arange(Wp, device=x.device).view(1,1,1,1,Wp)   # 0..W'-1
        dd   = torch.arange(D,  device=x.device).view(1,1,D,1,1)    # 0..D-1
        src  = (base - dd + L).clamp_(0, Wp-1).long()               # (1,1,D,1,W')
        xs   = torch.gather(x_pad, 4, src.expand(B,C,D,H,Wp))       # (B,C,D,H,W')
        return xs

    def inv_dw_transform_plus45(self, xs, W):
        # xs: (B,C,D,H,W') → (B,C,D,H,W)
        B,C,D,H,Wp = xs.shape
        L = D - 1
        base = torch.arange(W, device=xs.device).view(1,1,1,1,W)
        dd   = torch.arange(D, device=xs.device).view(1,1,D,1,1)
        src  = (base + dd).clamp_(0, Wp-1).long()
        xrec = torch.gather(xs, 4, src.expand(B,C,D,H,W))
        return xrec

    # 135° in (D,W): 使 w' = w - d   （每个 inline 切片 d 往 −W 方向平移 d 个像素）
    def dw_transform_135(self, x):
        # x: (B,C,D,H,W) → xs: (B,C,D,H,W'),   W' = W + (D-1)
        B,C,D,H,W = x.shape
        L, R = 0, (D-1)
        Wp = W + L + R
        x_pad = F.pad(x, (L, R, 0, 0, 0, 0))   # 这次右侧补 R

        base = torch.arange(Wp, device=x.device).view(1,1,1,1,Wp)
        dd   = torch.arange(D,  device=x.device).view(1,1,D,1,1)
        src  = (base + dd).clamp_(0, Wp-1).long()
        xs   = torch.gather(x_pad, 4, src.expand(B,C,D,H,Wp))
        return xs

    def inv_dw_transform_135(self, xs, W):
        B,C,D,H,Wp = xs.shape
        R = D - 1
        base = torch.arange(W, device=xs.device).view(1,1,1,1,W)
        dd   = torch.arange(D, device=xs.device).view(1,1,D,1,1)
        src  = (base - dd + R).clamp_(0, Wp-1).long()
        xrec = torch.gather(xs, 4, src.expand(B,C,D,H,W))
        return xrec


#####################################################   UP_dir
class UP_dir(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
        # self.upconv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        # self.conv = nn.Sequential(
        #     nn.Conv3d(in_channels,out_channels, kernel_size=1),
        #     nn.InstanceNorm3d(out_channels),nn.LeakyReLU(inplace=True),
        #     nn.Conv3d(out_channels ,out_channels, kernel_size=3, padding=1))
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels,out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1,out_channels),nn.LeakyReLU(negative_slope=0.01,inplace=True))        
        
        self.dirconv = DirectionalSurfaceConvBlock3D(in_channels,out_channels)

    def forward(self, x, encoder_features):
        x = self.up(x)
        diffZ = encoder_features.size()[2] - x.size()[2]
        diffY = encoder_features.size()[3] - x.size()[3]
        diffX = encoder_features.size()[4] - x.size()[4]
        x = nn.functional.pad(x, [diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2,
                                    diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x, encoder_features], dim=1)
        x0 = self.conv(x)
        x_dir = self.dirconv(x)

        return x0 + x_dir



#####################################################  UnetDecoderHead
class UnetDecoderHead(nn.Module):
    def __init__(self,input_feature_dims: list = [384, 192, 96, 48, 24],):
        super().__init__()
        self.deconv4 = DecoderBlock(input_feature_dims[0],input_feature_dims[1])
        self.deconv3 = DecoderBlock(input_feature_dims[1],input_feature_dims[2])
        self.deconv2 = DecoderBlock(input_feature_dims[2],input_feature_dims[3])
        self.deconv1 = DecoderBlock(input_feature_dims[3],input_feature_dims[4])

    def forward(self,c0, c1, c2, c3, c4):

        d3 = self.deconv4(c4,c3)
        d2 = self.deconv3(d3,c2)
        d1 = self.deconv2(d2,c1)
        d0 = self.deconv1(d1,c0)

        return d0
    
########################################################################    
class UnetDecoderHeadDir(nn.Module):
    def __init__(self,input_feature_dims: list = [384, 192, 96, 48, 24],):
        super().__init__()
        self.deconv4 = UP_dir(input_feature_dims[0],input_feature_dims[1])
        self.deconv3 = UP_dir(input_feature_dims[1],input_feature_dims[2])
        self.deconv2 = UP_dir(input_feature_dims[2],input_feature_dims[3])
        self.deconv1 = UP_dir(input_feature_dims[3],input_feature_dims[4])

    def forward(self,c0, c1, c2, c3, c4):

        d3 = self.deconv4(c4,c3)
        d2 = self.deconv3(d3,c2)
        d1 = self.deconv2(d2,c1)
        d0 = self.deconv1(d1,c0)

        return d0    

"""
"""
###############################################  MCF_Block
class MCF_Block(nn.Module):
    def __init__(self, in_channels, out_channels, rate=2):
        super(MCF_Block, self).__init__()
        expan_channels = out_channels * rate
        self.expansion_conv = nn.Conv3d(in_channels, expan_channels, 1)
        self.norm_conv = nn.Conv3d(expan_channels, expan_channels, 3, padding=1)
        self.segse_block = SegSEBlock(expan_channels)  # 
        self.zoom_conv = nn.Conv3d(expan_channels, out_channels, 1)
        self.skip_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.ln = nn.InstanceNorm3d(expan_channels) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x = self.relu(self.ln(self.expansion_conv(input)))  # in_c -- out_c*rate, k 1
        x = self.norm_conv(x)  # out_c*rate -- out_c*rate, k 3
        x = x * self.segse_block(x)  #  out_c*rate -- out_c*rate
        out = self.zoom_conv(x) + self.skip_conv(input)  # out 
        return out


##################################################  SegSEBlock
class SegSEBlock(nn.Module):
    def __init__(self, in_channels, rate=2):
        super().__init__()
        self.dila_conv = nn.Conv3d(in_channels, in_channels // rate, 3, padding=2, dilation=rate)
        self.conv1 = nn.Conv3d(in_channels // rate, in_channels, 1)
    def forward(self, input):
        x = self.dila_conv(input)
        x = self.conv1(x)
        x = nn.Sigmoid()(x)
        return x


###################################################  EdgeBlock
class EdgeBlock(nn.Module):
    def __init__(self, in_channels, out_channels,deconv_kernel):  
        super(EdgeBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)# , stride=1, padding=1)
        self.dconv = nn.ConvTranspose3d(out_channels, out_channels, 
                            kernel_size=deconv_kernel, stride=deconv_kernel, padding=0)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)# , stride=1, padding=1)
        self.ln = nn.InstanceNorm3d(1,out_channels) 
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.ln(self.conv1(x)))
        x = self.relu(self.ln(self.dconv(x)))
        out = self.relu(self.ln(self.conv2(x)))

        return out
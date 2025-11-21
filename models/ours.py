import torch.nn as nn
import torch
import torch.nn.functional as F


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

       
class ux_block(nn.Module):
    def __init__(self, dim): 
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = nn.GroupNorm(1,dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1, groups=dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1, groups=dim) 

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)  
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + x
        return x


class SCA3D(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel))
        self.spatial_se = nn.Conv3d(channel, 1, kernel_size=1,
                                    stride=1, padding=0, bias=False) 

    def forward(self, x):
        bahs, chs, _, _, _ = x.size()
        chn_se = self.avg_pool(x).view(bahs, chs)  
        chn_se = torch.sigmoid(self.channel_excitation(chn_se).view(bahs, chs, 1, 1,1))
        chn_se = torch.mul(x, chn_se)  
        spa_se = torch.sigmoid(self.spatial_se(x)) 
        spa_se = torch.mul(x, spa_se)  
        net_out = spa_se + x + chn_se
        return net_out


class DirectionalSurfaceConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(DirectionalSurfaceConvBlock3D, self).__init__()

        mid_channels = in_channels // 2  # 稍微加大一点中间通道，提升表达力
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1)
        self.ln1 = nn.GroupNorm(1, mid_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # 四个方向的卷积
        self.conv0 = nn.Conv3d(mid_channels, mid_channels // 4, kernel_size=(1, 9, 9), 
                               padding=(0, 4 * dilation_rate, 4 * dilation_rate), dilation=(1, dilation_rate, dilation_rate))
        self.conv90 = nn.Conv3d(mid_channels, mid_channels // 4, kernel_size=(9, 1, 9), 
                                padding=(4 * dilation_rate, 0, 4 * dilation_rate), dilation=(dilation_rate, 1, dilation_rate))
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
        x45 = self.inv_h_transform(self.conv45(self.h_transform(x)))
        x135 = self.inv_v_transform(self.conv135(self.v_transform(x)))

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

    def h_transform(self, x):
        """
        沿 W(X) 方向拉伸，模拟 45° 面
        """
        B, C, D, H, W = x.shape
        x = x.permute(0,1,3,2,4).contiguous()  # (B, C, H, D, W)
        x = x.view(B, C, H*D, W)
        x = torch.nn.functional.pad(x, (0, W-1))  # pad宽度方向
        x = x.view(B, C, H, D, 2*W-1)
        x = x.permute(0,1,3,2,4).contiguous()  # (B, C, D, H, newW)
        return x

    def inv_h_transform(self, x):
        """
        逆向 h_transform
        """
        B, C, D, H, W = x.shape
        x = x.permute(0,1,3,2,4).contiguous()  # (B, C, H, D, W)
        x = x.view(B, C, H*D, W)
        x = x[..., : (W + 1) // 2]  # 只取前一半
        x = x.view(B, C, H, D, (W + 1)//2)
        x = x.permute(0,1,3,2,4).contiguous()  # (B, C, D, H, W)
        return x

    def v_transform(self, x):
        """
        沿 H(Y) 方向拉伸，模拟 135° 面
        """
        B, C, D, H, W = x.shape
        x = x.permute(0,1,4,2,3).contiguous()  # (B, C, W, D, H)
        x = x.view(B, C, W*D, H)
        x = torch.nn.functional.pad(x, (0, H-1))  # pad高度方向
        x = x.view(B, C, W, D, 2*H-1)
        x = x.permute(0,1,3,4,2).contiguous()  # (B, C, D, H, newW)
        return x

    def inv_v_transform(self, x):
        """
        逆向 v_transform
        """
        B, C, D, H, W = x.shape
        x = x.permute(0,1,4,2,3).contiguous()  # (B, C, W, D, H)
        x = x.view(B, C, W*D, H)
        x = x[..., : (H + 1) // 2]
        x = x.view(B, C, W, D, (H + 1)//2)
        x = x.permute(0,1,3,4,2).contiguous()  # (B, C, D, H, W)
        return x

class DirectionalSurfaceConvBlock3D_0(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rate=1):
        super(DirectionalSurfaceConvBlock3D_0, self).__init__()

        mid_channels = in_channels // 2  
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1)
        self.ln1 = nn.GroupNorm(1, mid_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        self.conv0 = nn.Conv3d(mid_channels, mid_channels // 4, kernel_size=(1, 9, 9), 
                               padding=(0, 4 * dilation_rate, 4 * dilation_rate), dilation=(1, dilation_rate, dilation_rate))                     
        self.conv90 = nn.Conv3d(mid_channels, mid_channels // 4, kernel_size=(9, 1, 9), 
                                padding=(4 * dilation_rate, 0, 4 * dilation_rate), dilation=(dilation_rate, 1, dilation_rate))               
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
        B,C,D,H,W = x.shape
        L, R = (D-1), 0
        Wp = W + L + R
        x_pad = F.pad(x, (L, R, 0, 0, 0, 0))   
        base = torch.arange(Wp, device=x.device).view(1,1,1,1,Wp)   
        dd   = torch.arange(D,  device=x.device).view(1,1,D,1,1)    
        src  = (base - dd + L).clamp_(0, Wp-1).long()               
        xs   = torch.gather(x_pad, 4, src.expand(B,C,D,H,Wp))       
        return xs

    def inv_dw_transform_plus45(self, xs, W):
        B,C,D,H,Wp = xs.shape
        L = D - 1
        base = torch.arange(W, device=xs.device).view(1,1,1,1,W)
        dd   = torch.arange(D, device=xs.device).view(1,1,D,1,1)
        src  = (base + dd).clamp_(0, Wp-1).long()
        xrec = torch.gather(xs, 4, src.expand(B,C,D,H,W))
        return xrec


    def dw_transform_135(self, x):
        B,C,D,H,W = x.shape
        L, R = 0, (D-1)
        Wp = W + L + R
        x_pad = F.pad(x, (L, R, 0, 0, 0, 0))   
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


class UP_dir(nn.Module):
    def __init__(self, in_channels,out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)
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
        x = self.relu(self.ln(self.expansion_conv(input)))  
        x = self.norm_conv(x) 
        x = x * self.segse_block(x) 
        out = self.zoom_conv(x) + self.skip_conv(input) 
        return out

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


class EdgeBlock(nn.Module):
    def __init__(self, in_channels, out_channels,deconv_kernel):  
        super(EdgeBlock, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.dconv = nn.ConvTranspose3d(out_channels, out_channels, 
                            kernel_size=deconv_kernel, stride=deconv_kernel, padding=0)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.ln = nn.InstanceNorm3d(1,out_channels) 
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.ln(self.conv1(x)))
        x = self.relu(self.ln(self.dconv(x)))
        out = self.relu(self.ln(self.conv2(x)))

        return out




class hedupBlock(nn.Module):
    def __init__(self, in_channels, out_channels,up_sacle):  
        super(hedupBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.ln = nn.InstanceNorm3d(1,out_channels) 
        self.relu = nn.LeakyReLU(inplace=True)
        self.upsample_volume = nn.Upsample(scale_factor=(up_sacle, up_sacle, up_sacle), mode='trilinear', align_corners=True)

    def forward(self, x):
        x = self.relu(self.ln(self.conv1(x)))
        out = self.upsample_volume(x)
        return out
    
class DecoderHED(nn.Module):
    def __init__(self, input_feature_dims: list = [256,128,64,32,16],):
        super(DecoderHED, self).__init__()
        self.hedup_1 = hedupBlock(input_feature_dims[0],input_feature_dims[4], 16)
        self.hedup_2 = hedupBlock(input_feature_dims[1],input_feature_dims[4], 8)
        self.hedup_3 = hedupBlock(input_feature_dims[2],input_feature_dims[4], 4)
        self.hedup_4 = hedupBlock(input_feature_dims[3],input_feature_dims[4],2)

        self.conv_mix = nn.Sequential(nn.Conv3d(input_feature_dims[4]*5, input_feature_dims[4], 1),
                                      nn.InstanceNorm3d(input_feature_dims[4]),nn.LeakyReLU(inplace=True))
        self.conv = nn.Sequential(nn.Conv3d(input_feature_dims[4], 1, 1))

    def forward(self,c0, c1, c2, c3, c4):
        e4 = self.hedup_4(c1)  
        e3 = self.hedup_3(c2)  
        e2 = self.hedup_2(c3)  
        e1 = self.hedup_1(c4)
        d = torch.cat([c0,e1,e2,e3,e4], dim=1)
        d = self.conv_mix(d)
        out = self.conv(d)
        return out

class net_lower(nn.Module):
    def __init__(self,in_channels=1):
        super(net_lower, self ).__init__()
        self.inconv = nn.Sequential(nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
                                 nn.Conv3d(16, 16, kernel_size=3, padding=1),
                    nn.InstanceNorm3d(16),nn.LeakyReLU(inplace=True))   
        block= ux_block
        self.piexl_1 = self._make_piexl_layer_uxnet(block,16, 32, 2)  
        self.piexl_2 = self._make_piexl_layer_uxnet(block, 32, 64, 2)
        self.piexl_3 = self._make_piexl_layer_uxnet(block, 64, 128, 2)
        self.piexl_4 = self._make_piexl_layer_uxnet(block, 128, 256, 2)     

        self.decoder = DecoderHED()
        self.sigmoid = nn.Sigmoid()

    def _make_piexl_layer_uxnet_1(self, block, in_c, out_c,num_block):
        layers = []
        downsample = nn.Sequential(nn.GroupNorm(1,in_c),
                            nn.Conv3d(in_c, out_c, kernel_size=7, stride=2, padding=3),)
        layers.append(downsample)
        layers.append(block(out_c))  
        for _ in range(1, num_block):
            layers.append(block(out_c))
        return nn.Sequential(*layers)

    def _make_piexl_layer_uxnet(self, block, in_c, out_c,num_block):
        layers = []
        downsample = nn.Sequential(nn.GroupNorm(1,in_c),
                            nn.Conv3d(in_c, out_c, kernel_size=1, stride=2, bias=False))
        layers.append(downsample)
        layers.append(block(out_c))  
        for _ in range(1, num_block):
            layers.append(block(out_c))
        return nn.Sequential(*layers)  

    def forward(self, x):
        x0 = self.inconv(x)    
        x1 = self.piexl_1(x0) 
        x2 = self.piexl_2(x1) 
        x3 = self.piexl_3(x2) 
        x4 = self.piexl_4(x3) 

        out = self.decoder(x0, x1, x2, x3, x4)

        return self.sigmoid(out) 


class OutputBlock(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(OutputBlock, self).__init__()
        self.cbam = SCA3D(in_channels,in_channels)
        self.conv = nn.Conv3d(in_channels, n_classes, kernel_size=1)

    def forward(self, x):
        x = self.cbam(x)
        x =  self.conv(x)
        return x

        
class net_high(nn.Module):  
    """第二阶段网络：高分辨率细节分割"""
    def __init__(self, in_channels=1, n_classes=1):
        super(net_high, self).__init__()

        self.inconv = DoubleConv(in_channels,16)
        block= ux_block
        self.piexl_1 = self._make_piexl_layer_uxnet_1(block,16, 32, 2)  
        self.piexl_2 = self._make_piexl_layer_uxnet(block, 32, 64, 2)
        self.piexl_3 = self._make_piexl_layer_uxnet(block, 64, 128, 2)
        self.piexl_4 = self._make_piexl_layer_uxnet(block, 128, 256, 2)

        self.bottom = DoubleConv(256,256)

        self.up4 = UP_dir(384, 128)
        self.up3 = UP_dir(192, 64)  # 
        self.up2 = UP_dir(96, 32)
        self.up1 = UP_dir(48, 16)

        self.edge_4 = EdgeBlock(128,16, deconv_kernel=8)
        self.edge_3 = EdgeBlock(64,16, deconv_kernel=4)
        self.edge_2 = EdgeBlock(32,16, deconv_kernel=2)

        self.mixdiff = MCF_Block(48,16)

        self.outconv = OutputBlock(32,1)
        self.sigmoid = nn.Sigmoid()


    def _make_piexl_layer_uxnet_1(self, block, in_c, out_c,num_block):
        layers = []
        downsample = nn.Sequential(nn.GroupNorm(1,in_c),
                            nn.Conv3d(in_c, out_c, kernel_size=7, stride=2, padding=3),)
        layers.append(downsample)
        layers.append(block(out_c))  
        for _ in range(1, num_block):
            layers.append(block(out_c))
        return nn.Sequential(*layers)

    def _make_piexl_layer_uxnet(self, block, in_c, out_c,num_block):
        layers = []
        downsample = nn.Sequential(nn.GroupNorm(1,in_c),
                            nn.Conv3d(in_c, out_c, kernel_size=1, stride=2, bias=False))
        layers.append(downsample)
        layers.append(block(out_c))  
        for _ in range(1, num_block):
            layers.append(block(out_c))
        return nn.Sequential(*layers)  
    

    def forward(self, x):
        # encoder部分
        x0 = self.inconv(x)   
        x1 = self.piexl_1(x0)  
        x2 = self.piexl_2(x1)  
        x3 = self.piexl_3(x2)  
        x4 = self.piexl_4(x3) 
        x4 = self.bottom(x4)
    
        d4 = self.up4(x4,x3)  
        d3 = self.up3(d4,x2)  
        d2 = self.up2(d3,x1)  
        d1 = self.up1(d2,x0)  

        e4 = self.edge_4(d4)  
        e3 = self.edge_3(d3)  
        e2 = self.edge_2(d2)  

        
        y = self.mixdiff(torch.cat([e2, e3, e4], dim=1))  
        out = self.outconv(torch.cat([d1, y], dim=1))

        return self.sigmoid(out)



class ours_nn(nn.Module):  
    def __init__(self,in_channels=1):
        super(ours_nn, self).__init__()

        self.lowres_model = net_lower() 
        self.highres_model = net_high(in_channels=2)

    def forward(self, x):

        x_lowres = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True)
        lowres_output = self.lowres_model(x_lowres)
        lowres_seg_upsampled = F.interpolate(lowres_output, size=x.shape[2:], mode='trilinear', align_corners=True)
        

        combined_input = torch.cat([x, lowres_seg_upsampled], dim=1)
        highres_output = self.highres_model(combined_input)

        return highres_output , lowres_seg_upsampled


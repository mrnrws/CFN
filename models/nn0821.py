from models.ours.Block import ux_block,DoubleConv,UP_dir,EdgeBlock,MCF_Block,SCA3D
# from ours.Block import ux_block,DoubleConv,UP_dir,EdgeBlock,MCF_Block,SCA3D

import torch.nn as nn
import torch
import torch.nn.functional as F




class hedupBlock(nn.Module):
    def __init__(self, in_channels, out_channels,up_sacle):  
        super(hedupBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1) # , stride=1, padding=1)
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
        # print(c0.size(),c1.size(),c2.size(),c3.size(),c4.size())
        # torch.Size([1, 16, 128, 128, 128]) torch.Size([1, 32, 64, 64, 64]) torch.Size([1, 64, 32, 32, 32]) 
        # torch.Size([1, 128, 16, 16, 16]) torch.Size([1, 256, 8, 8, 8])
        e4 = self.hedup_4(c1)  
        e3 = self.hedup_3(c2)  
        e2 = self.hedup_2(c3)  
        e1 = self.hedup_1(c4)
        # print(e4.size(),e3.size(),e2.size(),e1.size())
        d = torch.cat([c0,e1,e2,e3,e4], dim=1)
        # print(d.size())
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
        self.piexl_1 = self._make_piexl_layer_uxnet(block,16, 32, 2)  # base_features=16
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
        x0 = self.inconv(x)    # x0 torch.Size([1, 16,  128])
        x1 = self.piexl_1(x0)  # x1 torch.Size([1, 32,  64, 64, 64])
        x2 = self.piexl_2(x1)  # x2 torch.Size([1, 64,  32, 32, 32])
        x3 = self.piexl_3(x2)  # x3 torch.Size([1, 128, 16, 16, 16])
        x4 = self.piexl_4(x3)  # x4 torch.Size([1, 256, 8, 8, 8])

        out = self.decoder(x0, x1, x2, x3, x4)

        return self.sigmoid(out) # self.sigmoid(d)   # torch.Size([1, 24, 64, 64, 64])


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
        self.piexl_1 = self._make_piexl_layer_uxnet_1(block,16, 32, 2)  # base_features=16
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
        x0 = self.inconv(x)   # x0 torch.Size([1, 16, 128, 128, 128])
        x1 = self.piexl_1(x0)  # x1 torch.Size([1, 32,  64, 64, 64])
        x2 = self.piexl_2(x1)  # x2 torch.Size([1, 64,  32, 32, 32])
        x3 = self.piexl_3(x2)  # x3 torch.Size([1, 128, 16, 16, 16])
        x4 = self.piexl_4(x3)  # x4 torch.Size([1, 256, 8, 8, 8])
        x4 = self.bottom(x4)
        # print('x4',x4.size())
        # print('x3',x3.size())
        # print('x2',x2.size())
        # print('x1',x1.size())
        # print('x0',x0.size())
    
        d4 = self.up4(x4,x3)  # # d3 torch.Size([1, 128, 16, 16, 16])
        d3 = self.up3(d4,x2)  # d3 torch.Size([1, 64, 32, 32, 32])
        d2 = self.up2(d3,x1)  # d2 torch.Size([1, 32, 64, 64, 64])
        d1 = self.up1(d2,x0)  # d1 torch.Size([1, 16, 128, 128, 128])
        
        e4 = self.edge_4(d4)  # e4 torch.Size([1, 16, 128, 128, 128])
        e3 = self.edge_3(d3)  # e3 torch.Size([1, 16, 128, 128, 128])
        e2 = self.edge_2(d2)  # e2 torch.Size([1, 16, 128, 128, 128])

        
        y = self.mixdiff(torch.cat([e2, e3, e4], dim=1))  # e5 torch.Size([1, 16, 128, 128, 128])
        out = self.outconv(torch.cat([d1, y], dim=1))

        return self.sigmoid(out)



class ours_nn_uxnet_0821(nn.Module):  
    def __init__(self,in_channels=1):
        super(ours_nn_uxnet_0821, self).__init__()

        self.lowres_model = net_lower() 
        self.highres_model = net_high(in_channels=2)

    def forward(self, x):

        x_lowres = F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=True)
        lowres_output = self.lowres_model(x_lowres)
        lowres_seg_upsampled = F.interpolate(lowres_output, size=x.shape[2:], mode='trilinear', align_corners=True)
        

        combined_input = torch.cat([x, lowres_seg_upsampled], dim=1)
        highres_output = self.highres_model(combined_input)

        return highres_output , lowres_seg_upsampled


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.randn((1, 1, 128,128,128), dtype=torch.float32).to(device)

    # model =  net_high(in_channels=1).to(device)   # False
    # result = model(data) 
    # print(result.size())

    model =  ours_nn_uxnet_0821().to(device)
    result, lowoutput = model(data) 
    print(result.size(),lowoutput.size())


    import thop
    flops, params = thop.profile(model, inputs=(data,))
    print("flops = {:.4f} G".format(flops / 1024.0 / 1024.0 / 1024.0))
    print("params = {:.4f} M".format(params / 1024.0 / 1024.0))

    # net_lower(in_channels=1)   24.6933 G   0.3840 M
    # net_high(in_channels=1)  784.7133 G    9.6964 M
    # ours_nn_uxnet_0821()   788.6437 G     10.0808 M

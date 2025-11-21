import torch
import torch.nn as nn




class Loss(nn.modules.loss._Loss):
    def __init__(self, args):
        super(Loss, self).__init__()
        print('Preparing loss function:')
        self.loss = []
        self.loss_module = nn.ModuleList()
        self.loss_names = []
        self.n_GPUs = args.n_GPUs
        print(f"args.loss = {args.loss}")

        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            loss_type = loss_type.strip()
            if loss_type == 'dice':
                loss_function = DiceLoss()
            elif loss_type == 'focal':
                loss_function = FocalLoss(gamma=2, alpha=0.9)  # 2,  0.9
            elif loss_type == 'maskdice':
                loss_function = MaskDice()
            elif loss_type == 'bce':
                loss_function = nn.BCELoss()
            elif loss_type =='Bbce':
                loss_function = BalancedBCELoss()
            elif loss_type == 'msssim':
                loss_function = MSSSIM()
            elif loss_type == 'smooth':
                loss_function = SmoothnessLoss()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            self.loss.append({'type': loss_type,'weight': float(weight),'function': loss_function})
            self.loss_names.append(loss_type)
            self.loss_module.append(loss_function)

        print(f"Registered loss types: {[l['type'] for l in self.loss]}")
        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(self.loss_module, range(args.n_GPUs) )

    def forward(self, sr, hr):
        losses = {}
        total_loss = 0
        for i, l in enumerate(self.loss):
            if l['type'] == 'dice':
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses[l['type']] = effective_loss.item()
            elif l['type'] == 'focal':
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses[l['type']] = effective_loss.item()
            elif l['type'] == 'maskdice':
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses[l['type']] = effective_loss.item()
            elif l['type'] == 'bce':
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses[l['type']] = effective_loss.item()
            elif l['type'] =='Bbce':
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses[l['type']] = effective_loss.item()
            elif l['type'] == 'msssim':
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses[l['type']] = effective_loss.item()
            elif l['type'] == 'smooth':
                loss = l['function'](sr)
                effective_loss = l['weight'] * loss
                losses[l['type']] = effective_loss.item()
            # else:
            #     raise ValueError(f"Unknown loss type: {l['type']}")
            total_loss += effective_loss
        
        return total_loss, losses
    

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module
        





'loss'
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean', epsilon=1e-6):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # 控制焦点的强度：较大 gamma 会强调难分类样本，较小 gamma 会让损失更接近传统交叉熵
        # 使得模型更加关注难以分类的样本，尤其是正类样本
        self.alpha = alpha  # 控制正负样本的平衡：当样本类别严重不平衡时，调整 alpha 可以给少数类样本更高的权重
        # 给正样本加权为 0.25，负样本的权重是 0.75
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, pred, target):
        pt = torch.where(target == 1, pred, 1 - pred)
        loss = -self.alpha * (1 - pt) ** self.gamma * torch.log(pt + self.epsilon)
        return loss.mean()  # 返回每个元素的损失


class  MaskDice(nn.Module):
    #  Tversky 系数
    def __init__(self,smooth=1,alpha = 0.7,reduction='mean'):
        super(MaskDice, self).__init__()
        self.smooth = smooth
        self.alpha = alpha  
        # alpha > 0.5 更关注 FN（适用于召回率更重要的任务）,alpha < 0.5 更关注 FP（适用于精准率更重要的任务）
        self.reduction = reduction
    def forward(self, predict, target):
        predict = predict.contiguous().view(predict.shape[0], -1)  # 展平成向量
        target = target.contiguous().view(target.shape[0], -1)
        
        num = torch.mul(predict, target)
        zeros = torch.zeros_like(num).float()
        true_pos = torch.sum(torch.where(target == -1., zeros, num), dim=1)  # 忽略区域
        
        false_neg = target * (1 - predict)
        false_neg = torch.sum(torch.where(target == -1., zeros, false_neg), dim=1)
        
        false_pos = (1 - target) * predict
        false_pos = torch.sum(torch.where(target == -1., zeros, false_pos), dim=1)
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        
        return (1 - tversky).mean()
        

class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
    def forward(self, pred, target):
        i = torch.sum(target)
        j = torch.sum(pred)
        intersection = torch.sum(target * pred)
        score = (2. * intersection + self.epsilon) / (i + j + self.epsilon)
        soft_dice =  1- score.mean()
        return soft_dice


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, pred):
        """
        pred: Tensor of shape (B, 1, D, H, W)
        """
        dz = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
        dy = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        dx = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])

        smooth_loss = (dx.mean() + dy.mean() + dz.mean())
        return smooth_loss


class BalancedBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        eps = 1e-7  # 避免 log(0) 计算错误
        num_pos = torch.sum(target == 1).float()
        num_neg = torch.sum(target == 0).float()
        pos_weight = (num_neg+eps) / (num_pos+eps)  
        loss = - (pos_weight * target * torch.log(pred + eps) + (1 - target) * torch.log(1 - pred + eps))
        return loss.mean()
    
' nn.BCELoss() 输入[0, 1]；BCEWithLogitsLoss() 输入(-∞, +∞)'   
'之前的都错了'

        

from math import exp
import torch.nn.functional as F

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        # TODO: store window between calls if possible
        return 1 - msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=True):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)
        img1 =  F.avg_pool3d(img1, kernel_size=2, stride=2)
        img2 =  F.avg_pool3d(img2, kernel_size=2, stride=2)
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2
    pow1 = mcs ** weights.view(-1, 1)
    pow2 = mssim ** weights.view(-1, 1)
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        max_val = 1 if torch.max(img1) <= 1 else 255
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range
    padd = 0
    (_, channel, D, H, W) = img1.size()
    if window is None:
        real_size = min(window_size, D, H, W)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv3d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv3d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv3d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
         ret = ssim_map.view(ssim_map.size(0), -1).mean(1)
    if full:
        return ret, cs
    return ret

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _3D_window = _1D_window @ _1D_window.t()
    _3D_window = _3D_window.unsqueeze(2) @ _1D_window.view(1, 1, -1)
    _3D_window = _3D_window.float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    window = _3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous()
    return window

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()
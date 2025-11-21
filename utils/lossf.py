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
            elif loss_type == 'bce':
                loss_function = nn.BCELoss()
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
            elif l['type'] == 'bce':
                loss = l['function'](sr, hr)
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




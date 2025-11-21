# 数据加载部分，transform 在这里统一设置为None，
# 数据增强在训练前单独完成，训练、验证、预测的数据都是经过(x-mean)/std正则化后的;
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
from skimage import transform


def normalization(data):
    data = data - np.mean(data)
    data = data / np.std(data)
    return data
# def normalization(x):  # 改了 gamma 变换 **
#     # Normalize the data to [0, 1] (assuming x has values in [0, 255] range)
#     return (x - np.min(x)) / (np.max(x) - np.min(x))


'200-3d'
def randomCrop3d(data, target,size=(128, 128, 128)):
    shape = data.shape
    size = np.array(size)
    lim = shape - size
    w = random.randint(0, lim[0])
    h = random.randint(0, lim[1])
    c = random.randint(0, lim[2])
    return data[w:w + size[0], h:h + size[1], c:c + size[2]], \
        target[w:w + size[0], h:h + size[1],c:c + size[2]]


def random_mask_seis(seis, cube_max_size=6, cube_max_num=10, line_max_width=4, line_max_num=15, enable_none=True):
    """
    对 seis 添加随机遮挡增强（cube、line 或 none）
    - seis: torch.Tensor, shape [1, D, H, W]
    - enable_none: 是否允许完全不遮挡（True 更符合 data augmentation 的概率性增强）
    """
    assert seis.ndim == 4, "seis must be [1, D, H, W]"
    C, D, H, W = seis.shape
    mask = torch.ones_like(seis)

    # ---- 三选一：cube、line、none ----
    if enable_none:
        mode = random.choice(['cube', 'line', 'none'])  # 概率各 1/3
    else:
        mode = random.choice(['cube', 'line'])          # 必定遮挡

    if mode == 'cube':
        cube_num = random.randint(0, cube_max_num)
        for _ in range(cube_num):
            size = random.randint(1, cube_max_size)
            dz = random.randint(0, max(D - size, 0))
            dy = random.randint(0, max(H - size, 0))
            dx = random.randint(0, max(W - size, 0))
            mask[:, dz:dz+size, dy:dy+size, dx:dx+size] = 0

    elif mode == 'line':
        line_num = random.randint(0, line_max_num)
        for _ in range(line_num):
            axis = random.choice([1, 2, 3])
            width = random.randint(1, line_max_width)
            if axis == 1:
                d = random.randint(0, max(D - width, 0))
                mask[:, d:d+width, :, :] = 0
            elif axis == 2:
                h = random.randint(0, max(H - width, 0))
                mask[:, :, h:h+width, :] = 0
            elif axis == 3:
                w = random.randint(0, max(W - width, 0))
                mask[:, :, :, w:w+width] = 0

    # mode == 'none' 时，mask 保持全1，即不遮挡
    return seis * mask



class FaultDataset3d(Dataset):
    def __init__(self, path, auge, kind, use_random_mask=True):
        self.path = path
        self.auge = auge
        self.use_random_mask = use_random_mask
        self.kind = kind
        self.dim = tuple([128]*3)
        self.image_list, self.label_list, self.id_list= self._load_datalist()

    def __getitem__(self, index):
        id = self.id_list[index]
        image, label = self._load_data(index)
        if self.auge == 'True' and self.kind =='train':
            image, label = self._augment_3d(image, label)

        x = np.expand_dims(image , axis=0)
        y = np.expand_dims(label, axis=0)
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x,y = torch.tensor(x, dtype=torch.float32) ,torch.tensor(y, dtype=torch.float32) 

        # if self.use_random_mask and self.kind == 'train':
        #     x = random_mask_seis(x, enable_none=True)
            
        return x,y,id

    def __len__(self):
        return len(self.image_list)
    
    def _augment_3d(self, image, label):
        gamma = np.clip(np.random.normal(loc=1, scale=0.2), a_min=0.65, a_max=1.38)
        image = image * gamma

        noise_scale = random.randint(50, 1200) / 10000.
        image += np.random.normal(loc=0, scale=noise_scale, size=image.shape)

        if random.random() < 0.5:
            scale1 = random.randint(128, 250)
            scale2 = random.randint(128, 200)
            scale3 = random.randint(128, 200)
            image = transform.resize(image, (scale1, scale2, scale3))
            label = (transform.resize(label, (scale1, scale2, scale3)) > 0.007).astype(np.int8)
            image, label = randomCrop3d(image, label)
        
        if random.random() < 0.5:
            image = np.flip(image, axis=0)  # 水平翻转, 沿 X 轴
            label = np.flip(label, axis=0)
        if random.random() < 0.5:
            image = np.flip(image, axis=1)  # 垂直翻转, 沿 Y 轴
            label = np.flip(label, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=2)  # 深度翻转, 沿 Z 轴
            label = np.flip(label, axis=2)
        return image, label

    def _load_data(self,index):
        image = np.fromfile(self.image_list[index], dtype=np.single).reshape(self.dim)
        label = np.fromfile(self.label_list[index], dtype=np.single).reshape(self.dim)
        return normalization(image), label

    def _load_datalist(self):
        img_list, label_list, id_list = [], [], []
        img_path = os.path.join(self.path, 'seis/')
        label_path = os.path.join(self.path, 'fault/')

        for item in os.listdir(img_path):
            img_list.append(os.path.join(img_path, item))
            label_list.append(os.path.join(label_path, item))
            id_list.append(item[:-4])
        return img_list, label_list,id_list



'field 128 volume'
# /root/data/datasets/FaultDataset/ours/shengli_cut_val
class FaultDatasetfieldshenglicut(Dataset):
    def __init__(self, path):
        self.path = path
        self.image_list, self.label_list, self.id_list= self._load_datalist()

    def __getitem__(self, index):
        id = self.id_list[index]
        image, label = self._load_data(index)

        x = np.expand_dims(image , axis=0)
        y = np.expand_dims(label, axis=0)
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        x,y = torch.tensor(x, dtype=torch.float32) ,torch.tensor(y, dtype=torch.float32) 
        return x,y,id
    
    def __len__(self):
        return len(self.image_list)
    
    def _load_data(self,index):
        image = np.load(self.image_list[index])
        label = np.load(self.label_list[index])
        return normalization(image), label

    def _load_datalist(self):
        img_list, label_list, id_list = [], [], []
        img_path = os.path.join(self.path,'seis/')  
        label_path = os.path.join(self.path,'label/')  
        for item in os.listdir(img_path):
            img_list.append(os.path.join(img_path, item))
            label_list.append(os.path.join(label_path, item))
            id_list.append(item[:-4])

        return img_list, label_list, id_list
    


# 没有用到
class FaultDataset3d_attr(Dataset):
    def __init__(self, path, auge,attr_mode,kind):
        self.path = path
        self.auge = auge
        self.kind = kind
        self.dim = tuple([128]*3)
        self.attr_mode = attr_mode
        self.image_list, self.label_list, self.coherence_list, self.ln_list,self.id_list= self._load_datalist()

    def __getitem__(self, index):
        id = self.id_list[index]
        image, label, coh, ln = self._load_data(index)
         
        if self.auge == 'True' and self.kind =='train':
            image, label, coh, ln = self._augment_3d(image, label, coh, ln)

        x = np.expand_dims(image , axis=0)
        y = np.expand_dims(label, axis=0)
        coh = np.expand_dims(coh, axis=0)
        ln = np.expand_dims(ln, axis=0)
        x = np.ascontiguousarray(x)
        y = np.ascontiguousarray(y)
        coh = np.ascontiguousarray(coh)
        ln = np.ascontiguousarray(ln)

        x,y = torch.tensor(x, dtype=torch.float32) ,torch.tensor(y, dtype=torch.float32) 
        coh,ln = torch.tensor(coh, dtype=torch.float32) ,torch.tensor(ln, dtype=torch.float32)
        if self.attr_mode == 'coherence':
            x = torch.cat([x, coh], dim=0)
        elif self.attr_mode == 'ln':
            x = torch.cat([x, ln], dim=0)
        elif self.attr_mode == 'both':
            x = torch.cat([x, coh, ln], dim=0)

        return x, y, id

    def __len__(self):
        return len(self.image_list)
    
    def _augment_3d(self, image, label,coh, ln):
        gamma = np.clip(np.random.normal(loc=1, scale=0.2), a_min=0.65, a_max=1.38)
        image = image * gamma

        noise_scale = random.randint(50, 1200) / 10000.
        image += np.random.normal(loc=0, scale=noise_scale, size=image.shape)

        if random.random() < 0.5:
            scale1 = random.randint(128, 250)
            scale2 = random.randint(128, 200)
            scale3 = random.randint(128, 200)
            image = transform.resize(image, (scale1, scale2, scale3))
            label = (transform.resize(label, (scale1, scale2, scale3)) > 0.007).astype(np.int8)
            coh = (transform.resize(coh, (scale1, scale2, scale3)) > 0.007).astype(np.int8)
            ln = (transform.resize(ln, (scale1, scale2, scale3)) > 0.007).astype(np.int8)
            image, label, coh, ln = randomCrop3d(image, label, coh, ln)
        
        if random.random() < 0.5:
            image = np.flip(image, axis=0)  # 水平翻转, 沿 X 轴
            label = np.flip(label, axis=0)
            coh = np.flip(coh, axis=0)
            ln = np.flip(ln, axis=0)
        if random.random() < 0.5:
            image = np.flip(image, axis=1)  # 垂直翻转, 沿 Y 轴
            label = np.flip(label, axis=1)
            coh = np.flip(coh, axis=1)
            ln = np.flip(ln, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=2)  # 深度翻转, 沿 Z 轴
            label = np.flip(label, axis=2)
            coh = np.flip(coh, axis=2)
            ln = np.flip(ln, axis=2)

        return image, label, coh, ln

    def _load_data(self,index):

        image = np.fromfile(self.image_list[index], dtype=np.single).reshape(self.dim)
        label = np.fromfile(self.label_list[index], dtype=np.single).reshape(self.dim)
        coherence = np.load(self.coherence_list[index])
        ln = np.load(self.ln_list[index])
        return normalization(image), label, normalization(coherence), normalization(ln)

    def _load_datalist(self):
        img_list, label_list,coherence_list,ln_list,id_list = [], [], [], [], []
        img_path = os.path.join(self.path,'seis/')
        label_path = os.path.join(self.path,'fault/')
        coherence_path = os.path.join(self.path,'attributes/coherence/')
        ln_path = os.path.join(self.path,'attributes/ln5/')

        for item in os.listdir(img_path):
            img_list.append(os.path.join(img_path, item))
            # 由于x和y的文件名一样，所以用一步加载进来
            label_list.append(os.path.join(label_path, item))
            attr_name = f'{item[:-4]}.npy'
            coherence_list.append(os.path.join(coherence_path,attr_name))
            ln_list.append(os.path.join(ln_path,attr_name))
            id_list.append(item[:-4])

        return img_list, label_list, coherence_list, ln_list, id_list


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    '3d'
    # dataset = FaultDataset3d_attr("/root/data/datasets/FaultDataset/train",True,attr_mode='both',kind ='train')
    
    dataset = FaultDataset3d("/home/user/data/zwt/datasets/FaultDataset/train",True,kind = 'train' )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True,num_workers=10, drop_last=True)
    
    print(len(dataset), ",  dataset created")  # 405
    print(len(dataloader), ",  dataloaders created")  # 105
    
    # for batch_datas, batch_labels, batch_attrs,filename in dataloader:
    #     print(batch_datas.size(),batch_labels.size(),batch_attrs.size(),filename)
    for i, data in enumerate(dataloader):
        inputs, labels, id = data
        print(i,id,inputs.size(),labels.size())   
    # torch.Size([4, 1, 128, 128]) torch.Size([4, 1, 128, 128])
    
    

    # # image, label,name = val_dataset.__getitem__(1)
    # # print(image.shape, label.shape,name)
    
    # # for batch_datas, batch_labels,filename in val_dataloader:
    # #     print(batch_datas.size(),batch_labels.size(),filename)
    # # for i, data in enumerate(val_dataloader):
    # #     inputs, labels, id = data
    # #     print(i,id,inputs.size(),labels.size())   # torch.Size([4, 1, 128, 128]) torch.Size([4, 1, 128, 128])   
    
import numpy as np
import os
import torch
from torch.utils.data import Dataset
import random
from skimage import transform


def normalization(data):
    data = data - np.mean(data)
    data = data / np.std(data)
    return data


def randomCrop3d(data, target,size=(128, 128, 128)):
    shape = data.shape
    size = np.array(size)
    lim = shape - size
    w = random.randint(0, lim[0])
    h = random.randint(0, lim[1])
    c = random.randint(0, lim[2])
    return data[w:w + size[0], h:h + size[1], c:c + size[2]], \
        target[w:w + size[0], h:h + size[1],c:c + size[2]]


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
            image = np.flip(image, axis=0)  
            label = np.flip(label, axis=0)
        if random.random() < 0.5:
            image = np.flip(image, axis=1)  
            label = np.flip(label, axis=1)
        if random.random() < 0.5:
            image = np.flip(image, axis=2)  
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



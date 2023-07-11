import os
import torch
import numpy as np
import random
from PIL import ImageFilter, ImageOps
from timm.data.readers.reader import Reader
from copy import deepcopy

class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

_TRAIN_SYNONYM = dict(train=None, training=None)
_EVAL_SYNONYM = dict(val=None, valid=None, validation=None, eval=None, evaluation=None)

def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split('[')[0]
    try_root = os.path.join(root, split_name)
    
    if os.path.exists(try_root):
        return try_root

    def _try(syn):
        for s in syn:
            try_root = os.path.join(root, s)
            if os.path.exists(try_root):
                return try_root
        return root
    if split_name in _TRAIN_SYNONYM:
        root = _try(_TRAIN_SYNONYM)
    elif split_name in _EVAL_SYNONYM:
        root = _try(_EVAL_SYNONYM)
    return root

'''
random apply torchvision transforms
'''
class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )
class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


# 对torchvision 和 albumentations的api做一个兼容
class TransformCompatWrapper():
    def __init__(self,transform):
        self.transform = transform

    def __call__(self,img):
        # return self.transform(img)
        try:
            # torchvision transforms
            return self.transform(img)
        except:
            # albumentations transforms
            return self.transform(image = np.asarray(img))['image']

# 实现K-FOLD验证的取数据
class ParserKFold(Reader):
    def __init__(
        self,
        parser,
        seed,
        kfold_now,
        kfold_all,
        mode='train',
        ):
        super().__init__()
        _rd_state = random.getstate()
        random.seed(seed+kfold_now)
        num = len(parser.samples) - int(len(parser.samples) * (1/kfold_all))
        samp = deepcopy(parser.samples)
        random.shuffle(samp)
        random.setstate(_rd_state)
        if mode == 'train':
            self.samples = samp[:num]
        else:
            self.samples = samp[num:]
            # print(self.samples)
    def __getitem__(self, index):
        path, target = self.samples[index]
        return open(path, 'rb'), target

    def __len__(self):
        return len(self.samples)

    def _filename(self, index, basename=False, absolute=False):
        filename = self.samples[index][0]
        if basename:
            filename = os.path.basename(filename)
        elif not absolute:
            filename = os.path.relpath(filename, self.root)
        return filename

def createDatasetWrapper(Dataset):
    class DatasetWrapper(Dataset):
        def __init__(self,target_transform=None,**kwargs) -> None:
            super().__init__(kwargs)
            self.target_transform = target_transform

        def __getitem__(self, index):
            img,target = super().__getitem__(index)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img,target
            
        def __iter__(self):
            img,target = super().__iter__()
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img,target

    return DatasetWrapper

class MultiViewWarper(object):
    def __init__(self, global_transo1,global_transo2=None,local_transo=None,local_crops_number=0):

        # first global crop
        self.global_transfo1 = global_transo1
        # second global crop
        self.global_transfo2 = global_transo2
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = local_transo

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        if self.local_transfo is not None:
            for _ in range(self.local_crops_number):
                crops.append(self.local_transfo(image))
        return crops
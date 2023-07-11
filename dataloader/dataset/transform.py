from timm.data import create_transform
from timm.data.transforms import str_to_interp_mode
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
import torch
import numpy as np
from .tranfg_autoda import AutoAugImageNetPolicy
from .transforms import *
from .utils import TransformCompatWrapper,MultiViewWarper

def _build_transform(config,is_train,_name):
    if _name == 'timm':
        _t = create_transform(
                        input_size=config.DATA.IMG_SIZE,
                        is_training=is_train,
                        use_prefetcher=config.DATA.TIMM_PREFETCHER,
                        no_aug = config.AUG.NO_AUG,
                        scale=config.AUG.SCALE,
                        ratio=config.AUG.RATIO,
                        hflip=config.AUG.HFLIP,
                        vflip=config.AUG.VFLIP,
                        color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                        auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                        interpolation=config.DATA.INTERPOLATION,
                        mean=config.AUG.NORM[0],
                        std=config.AUG.NORM[1],
                        re_prob=config.AUG.REPROB,
                        re_mode=config.AUG.REMODE,
                        re_count=config.AUG.RECOUNT,
                        crop_pct=config.TEST.CROP,
                        )

    elif _name == 'custom':
        tf_timm = create_transform(
                        input_size=config.DATA.IMG_SIZE,
                        is_training=is_train,
                        use_prefetcher=config.DATA.TIMM_PREFETCHER,
                        no_aug = config.AUG.NO_AUG,
                        scale=config.AUG.SCALE,
                        ratio=config.AUG.RATIO,
                        hflip=config.AUG.HFLIP,
                        vflip=config.AUG.VFLIP,
                        color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                        auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                        interpolation=config.DATA.INTERPOLATION,
                        mean=config.AUG.NORM[0],
                        std=config.AUG.NORM[1],
                        re_prob=config.AUG.REPROB,
                        re_mode=config.AUG.REMODE,
                        re_count=config.AUG.RECOUNT,
                        crop_pct=config.TEST.CROP,
                        separate=is_train,
                        )
        if is_train:
            [_tf1,tf2,tf3] = tf_timm
            if config.TEST.CROP != 1.0 and config.TEST.CROP != 0.:
                tf1 = [transforms.Resize(list((np.array(config.DATA.IMG_SIZE) / config.TEST.CROP).astype(int)), str_to_interp_mode(config.DATA.INTERPOLATION)),
                    transforms.RandomCrop(config.DATA.IMG_SIZE)]
            else:
                tf1 = [transforms.Resize(config.DATA.IMG_SIZE)]
            if config.AUG.HFLIP > 0:
                tf1 += [transforms.RandomHorizontalFlip(config.AUG.HFLIP)]
            if config.AUG.VFLIP:
                tf1 += [transforms.RandomVerticalFlip(config.AUG.VFLIP)]
            if config.AUG.TRANSFG_AA:
                tf1 += [AutoAugImageNetPolicy()]
            tf1 = tf1 + tf2.transforms + tf3.transforms
            _t = transforms.Compose(tf1)
        else:
            _t = tf_timm       

    elif _name == 'pim':
        # 448:600
        # 384:510
        # 768:
        if is_train:
            # transforms.RandomApply([RandAugment(n=2, m=3, img_size=data_size)], p=0.1)
            # RandAugment(n=2, m=3, img_size=sub_data_size)
            _t = transforms.Compose([
                        transforms.Resize(list((np.array(config.DATA.IMG_SIZE) / config.TEST.CROP).astype(int)), str_to_interp_mode(config.DATA.INTERPOLATION)),
                        transforms.RandomCrop(config.DATA.IMG_SIZE),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))], p=0.1),
                        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=torch.tensor(config.AUG.NORM[0]),
                            std=torch.tensor(config.AUG.NORM[1])
                        ),
                ])
        else:
            _t = transforms.Compose([
                        transforms.Resize(list((np.array(config.DATA.IMG_SIZE) / config.TEST.CROP).astype(int)), str_to_interp_mode(config.DATA.INTERPOLATION)),
                        transforms.CenterCrop(config.DATA.IMG_SIZE),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=torch.tensor(config.AUG.NORM[0]),
                            std=torch.tensor(config.AUG.NORM[1])
                        ),
                ]) 

    elif _name == 'pict':
        t1 = [A.Resize(height=config.DATA.IMG_SIZE[0],width=config.DATA.IMG_SIZE[1])]
        if (is_train and not config.AUG.NO_AUG) or (not is_train and config.AUG.TEST_AUG):
            # t2 = [
            #         A.RandomBrightnessContrast(p=0.7),
            #         A.HorizontalFlip(p=0.7),
            #         A.VerticalFlip(p=0.7),
            #         A.ShiftScaleRotate(rotate_limit=15.0, p=0.7),
            #         A.OneOf([
            #             A.Emboss(p=1),
            #             A.Sharpen(p=1),
            #             A.Blur(p=1)
            #                 ], p=0.7),
            #     ]
            t2 = [
                    A.RandomBrightnessContrast(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.ShiftScaleRotate(rotate_limit=15.0, p=0.5),
                ]
        else:
            if config.AUG.GRAY:
                t2=[A.ToGray(p=1)]
            else:
                t2 = []
        t3 = [A.Normalize(mean=config.AUG.NORM[0], std=config.AUG.NORM[1]),
            ToTensorV2()]

        _t = [A.Compose(t1),A.Compose(t2),A.Compose(t3)] if config.AUG.SEPARATE else A.Compose(t1+t2+t3)

    elif _name == 'simmim':
        _t = SimMIMTransform(config)

    elif _name == 'dino':
        _t = DataAugmentationDINO(config.AUG.DINO.GLOBAL_CROPS_SCALE,config.AUG.DINO.LOCAL_CROPS_SCALE,config.AUG.DINO.LOCAL_CROPS_NUMBER,config.DATA.IMG_SIZE[0])

    elif _name == 'dinofgia':
        if is_train:
            _t = DINOFGIA(config.AUG.SCALE,config.AUG.DINO.LOCAL_CROPS_SCALE,config.AUG.DINO.LOCAL_CROPS_NUMBER,config.DATA.IMG_SIZE[0])
        else:
            _t = transforms.Compose([
                    transforms.Resize(list((np.array(config.DATA.IMG_SIZE) / config.TEST.CROP).astype(int)), str_to_interp_mode(config.DATA.INTERPOLATION)),
                    transforms.CenterCrop(config.DATA.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(config.AUG.NORM[0]),
                        std=torch.tensor(config.AUG.NORM[1])
                    ),
            ])
            
    elif _name == 'multiviewpict':
        try:
            multi_view = config.PICT.MULTI_VIEW
        except:
            multi_view = 'strong_none'
        _t = PicTMultiViewTransform(is_train,type=multi_view,image_size=config.DATA.IMG_SIZE,mean_std=config.AUG.NORM)

    elif _name == 'multiviewfgia':
        if is_train:
            _t = MULTIVIEWFGIA(config.AUG.SCALE,config.AUG.DINO.LOCAL_CROPS_SCALE,config.AUG.DINO.LOCAL_CROPS_NUMBER,config.DATA.IMG_SIZE[0],multi_view=config.AUG.DINO.MULTI_VIEW)
        else:
            _t = transforms.Compose([
                    transforms.Resize(list((np.array(config.DATA.IMG_SIZE) / config.TEST.CROP).astype(int)), str_to_interp_mode(config.DATA.INTERPOLATION)),
                    transforms.CenterCrop(config.DATA.IMG_SIZE),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(config.AUG.NORM[0]),
                        std=torch.tensor(config.AUG.NORM[1])
                    ),
            ])

    elif _name == 'simsiam':
        _t = SimSiamTransform(config.DATA.IMG_SIZE[0],mean_std = config.AUG.NORM)

    else:
        raise NotImplementedError

    # Transform Compatible Wrapper, compat between torchvision.transforms and albumentations
    if type(_t) in (list,tuple):
        for i in range(len(_t)):
            _t[i] = TransformCompatWrapper(_t[i])
    else:
        _t = TransformCompatWrapper(_t)
   
    return _t

def _build_target_transform(config,is_train):
    if config.TARGET_AUG.TO_BIN_TARGET:
        t1 = [ToBinTarget(config.DATA.DATA_NOR_INDEX)]
    
    return transforms.Compose(t1)

def build_transform(config,is_train):
    if config.AUG.MULTI_VIEW is not None and is_train:
        # _type = config.AUG.MULTI_VIEW.lower().split('_')
        # assert all(_t in ('strong','weak','none') for _t in _type)
        # _transforms = ()
        # for _t in _type:
        #     _transforms += (_build_transform(config,is_train,_t),)

        # DINO multiview
        _names = config.DATA.DATALOADER_NAME.lower().split('_')[2]
        _names = _names.split('/')
        global_transo1 = _build_transform(config,is_train,_names[0])
        if len(_names)<2:
            _names.append(_names[0])
        global_transo2 = _build_transform(config,is_train,_names[1])
        local_transo2 = _build_transform(config,is_train,_names[2]) if len(_names)>2 else None
        _transforms = MultiViewWarper(global_transo1,global_transo2,local_transo2,local_crops_number=config.AUG.DINO.LOCAL_CROPS_NUMBER)
    else:
        _name = config.DATA.DATALOADER_NAME.lower().split('_')[2]
        _transforms = _build_transform(config,is_train,_name)
    if config.TARGET_AUG.NO_AUG:
        _transforms_target = None
    else:
        _transforms_target = _build_target_transform(config,is_train)
    return _transforms,_transforms_target

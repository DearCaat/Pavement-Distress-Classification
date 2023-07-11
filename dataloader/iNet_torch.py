from timm.data import create_loader
import torch
from .dataset import build_dataset

def timm_dataloader(config,is_train):
    if is_train:
        dataset_train,dataset_val = build_dataset(config,'train_val')

        loader_train = create_loader(
            dataset_train,
            input_size=config.DATA.IMG_SIZE,
            batch_size=config.DATA.BATCH_SIZE,
            is_training=True,
            use_prefetcher=config.DATA.TIMM_PREFETCHER,
            no_aug=config.AUG.NO_AUG,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            scale=config.AUG.SCALE,
            ratio=config.AUG.RATIO,
            hflip=config.AUG.HFLIP,
            vflip=config.AUG.VFLIP,
            color_jitter=config.AUG.COLOR_JITTER,
            auto_augment=config.AUG.AUTO_AUGMENT,
            num_aug_splits=config.AUG.SPLITS,
            interpolation=config.DATA.INTERPOLATION,
            mean=config.AUG.NORM[0],
            std=config.AUG.NORM[1],
            num_workers=config.DATA.NUM_WORKERS,
            distributed=config.DISTRIBUTED,
            pin_memory=config.DATA.PIN_MEMORY,
        )

        loader_val = create_loader(
            dataset_val,
            input_size=config.DATA.IMG_SIZE,
            batch_size=config.DATA.VAL_BATCH_SIZE,
            is_training=False,
            use_prefetcher=config.DATA.TIMM_PREFETCHER,
            interpolation=config.DATA.INTERPOLATION,
            mean=config.AUG.NORM[0],
            std=config.AUG.NORM[1],
            num_workers=config.DATA.NUM_WORKERS,
            distributed=config.DISTRIBUTED,
            crop_pct=config.TEST.CROP,
            pin_memory=config.DATA.PIN_MEMORY,
        )
        return dataset_train, dataset_val, loader_train, loader_val
    else:
        dataset_test = build_dataset(config,'test')

        loader_test = create_loader(
            dataset_test,
            input_size=config.DATA.IMG_SIZE,
            batch_size=config.DATA.VAL_BATCH_SIZE,
            is_training=False,
            use_prefetcher=config.DATA.TIMM_PREFETCHER,
            interpolation=config.DATA.INTERPOLATION,
            mean=config.AUG.NORM[0],
            std=config.AUG.NORM[1],
            num_workers=config.DATA.NUM_WORKERS,
            distributed=config.DISTRIBUTED,
            crop_pct=config.TEST.CROP,
            pin_memory=config.DATA.PIN_MEMORY,
        )
        return dataset_test,loader_test

def pytorch_dataloader(config,is_train):
    if is_train:
        dataset_train,dataset_val = build_dataset(config,'train_val')
        if config.DISTRIBUTED:
            sampler=torch.utils.data.distributed.DistributedSampler(dataset_train)
            loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.DATA.BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,drop_last=config.DATA.DROP_LAST,persistent_workers=True,sampler=sampler)
        else:
            loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=config.DATA.BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,drop_last=config.DATA.DROP_LAST,persistent_workers=True,shuffle=True)
        loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=config.DATA.VAL_BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,persistent_workers=True)
        
        return dataset_train, dataset_val, loader_train, loader_val
        
    else:
        dataset_test = build_dataset(config,'test')

        loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=config.DATA.VAL_BATCH_SIZE,num_workers=config.DATA.NUM_WORKERS,pin_memory=config.DATA.PIN_MEMORY,persistent_workers=True)
        return dataset_test,loader_test
from .transform import build_transform
from timm.data import create_dataset
from .utils import _search_split,ParserKFold 
from .datasets import *
from .fgvc import *
def _build_dataset(config,_type='train'):

    name = config.DATA.DATALOADER_NAME.lower().split('_')[1]
    
    if _type == 'train':
        split = config.DATA.TRAIN_SPLIT
        is_train = True
    elif _type == 'val':
        split = config.DATA.VAL_SPLIT
        is_train = False
    elif _type == 'test':
        split = config.DATA.TEST_SPLIT
        is_train = False

    if config.TEST.K_FOLD_VAL_ENABLE:
        split = config.DATA.TRAIN_SPLIT

    transform,target_transform = build_transform(config,is_train)

    # deal with the class map, this problem caused by yacs
    # convert the list to dict, the value of dict is the index
    class_map = config.DATA.CLASS_MAP or ''
    if isinstance(class_map,(tuple,list)):
        _class_map = {}
        for idx,_name in enumerate(class_map):
           _class_map.update({_name:idx}) 
        class_map = _class_map

    if name == 'img':
         
        dataset = ImageDataset(root=_search_split(config.DATA.DATA_PATH, split),transform=transform,target_transform=target_transform,class_map=class_map)

    elif name == 'timm':
        if config.DATA.DATALOADER_NAME.lower().split('_')[0] == 'timm':
            transform = None

        dataset = create_dataset(
        config.DATA.DATASET,
        root=config.DATA.DATA_PATH, split=split, is_training=is_train,
        batch_size=config.DATA.BATCH_SIZE,repeats=config.DATA.EPOCH_REPEATS,transform=transform,class_map=class_map,target_transform=target_transform)
    elif name == 'multiview':
        dataset = MulitiViewImageDataset(root=_search_split(config.DATA.DATA_PATH, split),transform=transform,is_multi_view=config.AUG.MULTI_VIEW,size=config.DATA.IMG_SIZE,timm_trans=config.AUG.TIMM_TRANS,target_transform=target_transform,class_map=class_map)
    elif name == 'patch':
        dataset = PatchImageDataset(root=_search_split(config.DATA.DATA_PATH, split),transform=transform,target_transform=target_transform,last_transform=config.DATA.LAST_TRANSFORM,is_ip=config.DATA.IS_IP,patch_size=config.DATA.PATCH_SIZE,stride=config.DATA.STRIDE,class_map=class_map)
    else:
        raise NotImplementedError('Dataset: '+name)
    if config.TEST.K_FOLD_VAL_ENABLE:
        dataset.parser = ParserKFold(dataset.parser,config.SEED,config.TEST.K_FOLD_VAL_NOW,config.TEST.K_FOLD_VAL_ALL,mode=_type)
        
    return dataset
def build_dataset(config,_type='train_val'):
    
    _type = _type.lower()
    _type = _type.split('_')

    assert all(_t in ('train','val','test') for _t in _type)

    datasets = ()

    for _t in _type:
        datasets += (_build_dataset(config,_t),)
    if len(datasets) == 1:
        datasets = datasets[0]
    return datasets

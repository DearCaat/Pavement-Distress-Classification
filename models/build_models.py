from timm.models import create_model
from .wsplin import *
from .ioplin import *
from .utils import LinearProbWrapper
from .pict import *

# from ._vit import *

ONE_BACKBONE_GROUP = ('wsplin','ioplin','pict')

def build_model(config,is_backbone=False,num_classes=None,logger=None):
    if is_backbone:
        model_name = config.MODEL.BACKBONE
        features_only = False
    else:
        model_name = config.MODEL.NAME.lower()
        features_only = False
    num_classes = num_classes or config.MODEL.NUM_CLASSES

    # Use the official impl to use the gradient cheackpoint
    if model_name.startswith('vgg'):
        model = create_model(
            model_name,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=num_classes,
            features_only=features_only
        )
        models = {'main':model}
    elif model_name.startswith(('res','incep')):
        model = create_model(
            model_name,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=num_classes,
            drop_rate=None if int(config.MODEL.DROP_RATE) == -1 else config.MODEL.DROP_RATE,
            drop_path_rate=None if int(config.MODEL.DROP_PATH_RATE) == -1 else config.MODEL.DROP_PATH_RATE,
            features_only=features_only
        )
        models = {'main':model}
    elif model_name.startswith('tf_effi'):
        model = create_model(
            model_name,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=num_classes,
            drop_rate=None if int(config.MODEL.DROP_RATE) == -1 else config.MODEL.DROP_RATE,
            drop_path_rate=None if int(config.MODEL.DROP_PATH_RATE) == -1 else config.MODEL.DROP_PATH_RATE,
            features_only=features_only
        )
        models = {'main':model}
    # The base framework which includes the one backbone
    elif model_name.startswith(ONE_BACKBONE_GROUP):
        backbone = build_model(config,is_backbone=True)['main']
        models = create_model(
            model_name,
            backbone = backbone,
            config = config,
            logger=logger
        )
    else:
        # drop_rate=config.MODEL.DROP_RATE,
        # drop_path_rate=config.MODEL.DROP_PATH_RATE
        model = create_model(
            model_name,
            pretrained=config.MODEL.PRETRAINED,
            num_classes=num_classes,
            drop_rate=None if int(config.MODEL.DROP_RATE) == -1 else config.MODEL.DROP_RATE,
            drop_path_rate=None if int(config.MODEL.DROP_PATH_RATE) == -1 else config.MODEL.DROP_PATH_RATE,
            img_size = config.DATA.IMG_SIZE[0],
            features_only=features_only
        )
        models = {'main':model}

    # 处理线性测试的情况
    if config.TEST.LINEAR_PROB.ENABLE and not is_backbone:
        models['main'] = LinearProbWrapper(models['main'],config)

    return models
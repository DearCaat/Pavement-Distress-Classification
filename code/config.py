import os
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# If not set the value, the default is the 3 times of BATCH_SIZE
_C.DATA.VAL_BATCH_SIZE = -1
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Path to pretrained model
_C.DATA.PRETRAINED_DIR = ''
# Dataset name     tfds/cqu_bpdd  cfd crack500 cracktre200
_C.DATA.DATASET = 'cqu_bpdd'

# 在类别中，正常图片的类别索引，该值判断在下面那个值之后
# 当二分类训练时，该值会决定类别中哪一个类为negative
_C.DATA.CLS_NOR_INDEX = 6
# 数据集中的正常图片所在的类别索引 cqu_bpdd ：6
_C.DATA.DATA_NOR_INDEX = 6
# 与前两个字段不同的是，该字段影响的是在dataloader阶段的对应。而前者是在dataloader之后的对应
_C.DATA.CLASS_MAP = None
# Input image size (h,w)  cqu_bpdd (900,1200) cfd(300,450)
_C.DATA.IMG_SIZE = (224,224)
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bilinear'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.TFRECORD_MODE = False
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Number of data loading threads
_C.DATA.NUM_WORKERS = 4
# dataset train split (default: train)
_C.DATA.TRAIN_SPLIT = 'train'
# dataset validation split (default: validation)
_C.DATA.VAL_SPLIT = 'val'
# dataset test split (default: test)
_C.DATA.TEST_SPLIT = 'test'
# epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).
_C.DATA.EPOCH_REPEATS = 0

_C.DATA.TIMM_PREFETCHER = False
# pytorch dataloader
# timm dataloader default True when set the is_training is True
_C.DATA.DROP_LAST = True
# "dataloader_dataset_transform"  
# transform={torch,timm,custom...}. The first three totally depends on the _C.AUG configs
# dataset = {img,timm,tfds,multiview...}. The first one is almost same with timm ImageDataset, but add the target transform.
_C.DATA.DATALOADER_NAME = 'timm_timm_timm'
# 数据集的大小，该值会在main函数中被覆写
_C.DATA.LEN_DATALOADER_TRAIN = 0
_C.DATA.LEN_DATALOADER_VAL = 0
_C.DATA.LEN_DATALOADER_TEST = 0
_C.DATA.LEN_DATASET_TRAIN = 0
_C.DATA.LEN_DATASET_VAL = 0
_C.DATA.LEN_DATASET_TEST = 0
# -----------------------------------------------------------------------------
# Trainer settings, more settings please refer to /configs/**.yaml
# -----------------------------------------------------------------------------
_C.TRAINER = CN()
_C.TRAINER.NAME = 'iNet_cls'

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type        #swin_small_patch4_window7_224  efficientnetv2_rw_s  deit_base_patch16_224  tf_efficientnet_b3 vit_base_patch32_224
_C.MODEL.NAME = 'pict_swin_small_patch4_window7_224'
# Model name
_C.MODEL.BACKBONE = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Only load the model status when resuming from the checkpoint
_C.MODEL.ONLY_LOAD_MODEL=False
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 8
# Dropout rate     effi-b3 0.3`
_C.MODEL.DROP_RATE = -1.
# Drop path rate   effi-b3 0.2  swin_s 0.3 swin_b_p4w12 0.5 -1让timm自己处理
_C.MODEL.DROP_PATH_RATE = -1.
# Label Smoothing
#_C.MODEL.LABEL_SMOOTHING = 0.1
_C.MODEL.LABEL_SMOOTHING = 0.
# Start with pretrained version of specified network (if avail)
_C.MODEL.PRETRAINED = True

# 在多模型训练的时候，到底哪些模型需要存放在GPU中，main始终是主模型，在最前面
_C.MODEL.TOGPU_MODEL_NAME = ['main'] 
# 在多模型训练的时候，到底哪些模型需要保存在checkpoint中，,除main以外
_C.MODEL.SAVE_OTHER_MODEL_NAME = [] 
# 在多模型训练的时候，到底哪些模型需要保留下最佳模型中，main始终是主模型，在最前面,每一个都会单独存一个文件，训练过程中也会有单独的ckpt文件
_C.MODEL.SAVE_BEST_MODEL_NAME = ['main'] 

# Torch 2.0 Model Compile  default, reduce-overhead (extra memo) or max-autotune (more compile time)
_C.MODEL.COMPILER_MODE = 'default'


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 20
_C.TRAIN.WARMUP_EPOCHS = 3
# 更高的优先级相较于EPOCHS来说
_C.TRAIN.WARMUP_STEPS = -1
_C.TRAIN.WEIGHT_DECAY = .0
_C.TRAIN.BASE_LR = 1e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-7
# Clip gradient norm                                     
#_C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.CLIP_GRAD = 0.
# Gradient clipping mode. One of ("norm", "value", "agc")
_C.TRAIN.CLIP_MODE = 'norm'
# LR batch size scale Default is 512, if the value == batch_size or <= 0, no scale is employed
_C.TRAIN.LR_BS_SCALE = 512
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument 
# Default is 1. new ver. 20220523
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

_C.TRAIN.NO_VAL = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'  #'cosine'
# 该字段要求用户自定义优化器参数分组函数，该字段不能直接使用。e.g. simsiam_param_grouping
_C.TRAIN.LR_SCHEDULER.CONSTANT_LR_FIELD = None
# STEP interval to decay LR, used in flat_cosine
_C.TRAIN.LR_SCHEDULER.DECAY_STEPS_RATIO=0.5
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 2
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
 #'Lookahead_adamw'
_C.TRAIN.OPTIMIZER.NAME =  'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.OPTIMIZER.PARAM_GROUPS_FUNC = None
# Accurate lr scale setting, default is true
_C.TRAIN.AS_LR_SCALE = True
#Loss
_C.TRAIN.LOSS = CN()
# support multi-loss: loss1_loss2_loss3
_C.TRAIN.LOSS.NAME = 'crossentropy'

# -----------------------------------------------------------------------------
# Target Augmentation settings
# -----------------------------------------------------------------------------
_C.TARGET_AUG = CN()
# same with AUG.NO_AUG
_C.TARGET_AUG.NO_AUG = True
# whether 
_C.TARGET_AUG.TO_BIN_TARGET = False

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Norm mean and std, default is [IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD], but old wsplin model use [(0.455,0.455,0.455),(0.225,0.225,0.225)]  effi-b3 use [(0.5,0.5,0.5),(0.5,0.5,0.5)]
_C.AUG.NORM = [IMAGENET_DEFAULT_MEAN,IMAGENET_DEFAULT_STD]
# Disable all training augmentation, override other train aug args
_C.AUG.NO_AUG = True
# 
_C.AUG.SEPARATE = False
# Convert the img to gray
_C.AUG.GRAY = False
# apply the data augument in test phase
_C.AUG.TEST_AUG = False
# Random resize scale (default: 0.08 1.0)
_C.AUG.SCALE = [0.08, 1.0]
# Random resize aspect ratio (default: 0.75 1.33)
_C.AUG.RATIO = [3./4., 4./3.]
# Horizontal flip training aug probability
_C.AUG.HFLIP = 0.5
# Vertical flip training aug probability
_C.AUG.VFLIP = 0.
# Color jitter factor   0.4
_C.AUG.COLOR_JITTER = 0.  
# Use AutoAugment policy. "v0" or "original" rand-m3-n2-mstd0.5
_C.AUG.AUTO_AUGMENT = None
# Random erase prob
_C.AUG.REPROB = 0.   #0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
'''# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0

# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'''
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 0.
# Number of augmentation splits (default: 0, valid: 0 or >=2)
_C.AUG.SPLITS=0
# output multi-view images, "strong_weak","strong_none","weak_none"
# "student_teacher"
_C.AUG.MULTI_VIEW = None

# 不同于timm的AA，这是transfg那篇文章所用的aa
_C.AUG.TRANSFG_AA = False
# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.TOPK = (1,5)
# Whether to use center crop when testing
_C.TEST.CROP = 1.
# acc1 macro_f1 auc，['model_best_save_idx','metric']
# Min:开头用于指明，该指标越小越好
_C.TEST.BEST_MODEL_METRIC = ['main','acc1']
# 这里面的metric会以最小值作为best value
_C.TEST.MIN_BEST_METRIC =  []
# save the last epoch model, independent of the BEST_MODEL_METRIC.
_C.TEST.SAVE_LAST_MODEL = False
# 二分类测试，该值会决定在多分类训练或者标签本身是多类别时，是否还进行二分类测试
_C.TEST.BINARY_MODE = False
# Linear prob, for SSL
_C.TEST.LINEAR_PROB = CN() 
_C.TEST.LINEAR_PROB.ENABLE = False
_C.TEST.LINEAR_PROB.DIM = 1024
# KNN, for SSL
_C.TEST.KNN = False
# K Fold validation
_C.TEST.K_FOLD_VAL_ENABLE = False
_C.TEST.K_FOLD_VAL_ALL = 5
_C.TEST.K_FOLD_VAL_NOW = 1
# 训练多少次，默认是跟折数一直
_C.TEST.K_FOLD_VAL_TIMES = 5
# P@R
_C.TEST.PR = False
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# use NVIDIA Apex AMP or Native AMP for mixed precision training
# overwritten by command line argument
_C.AMP = True
# Use NVIDIA Apex AMP mixed precision default O1
_C.APEX_AMP = False
# Use Native Torch AMP mixed precision
_C.NATIVE_AMP = True
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# name of experiment, overwritten by command line argument
_C.EXP_NAME = "default"
# name of project, overwritten by command line argument
_C.PROJECT_NAME = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 50
# Fixed random seed
_C.SEED = 42

# torch compile
_C.NO_COMPILE = False

# the dir of tested data
_C.LOAD_TEST_DIR = ''
# log training and validation metrics to wandb
_C.LOG_WANDB = False
_C.LOG_WANDB_WATCH = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
_C.DISTRIBUTED = False
_C.WORLD_SIZE = 1
# apply SyncBN in DDP
_C.SYNCBN = True

_C.MODEL_EMA = False
_C.EMA_FORCE_CPU = False
_C.EMA_DECAY = 0.9996
_C.EMA_SCHEDULER = None

_C.NO_TRAIN = False

_C.TRAIN_MODE = 't_e'
_C.EMPTY_CACHE = False

def _update_config_from_file(config, cfg_file):
    config.defrost()
    config.set_new_allowed(True)
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    if args.trainer:
        config.TRAINER.NAME = args.trainer
    # pth = os.path.join(os.path.abspath('.'),'configs',config.TRAINER.NAME+'.yaml')
    # _update_config_from_file(config, pth)

    if args.cfg:
        if type(args.cfg) in (tuple,list):
            for _cfg in args.cfg:
                _update_config_from_file(config, _cfg)
        else:
            _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        _opts = []
        # 将 ['key=value']转换成['key','value']
        if '=' in args.opts[0]:
            for opt in args.opts:
                k,v = opt.split('=')
                _opts += [k,v]
        else:
            _opts = args.opts
        config.merge_from_list(_opts)

    # merge from specific arguments
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.no_val:
        config.TRAIN.NO_VAL = True
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if config.DATA.VAL_BATCH_SIZE == -1:
        config.DATA.VAL_BATCH_SIZE = 3 * config.DATA.BATCH_SIZE
    if args.val_batch_size:
        config.DATA.VAL_BATCH_SIZE = args.val_batch_size
    if args.log_wandb:
        config.LOG_WANDB = args.log_wandb
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.tfrecord:
        config.DATA.TFRECORD_MODE = True
    if args.title:
        config.EXP_NAME = args.title
    if args.project:
        config.PROJECT_NAME = args.project
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.no_amp:
        config.AMP = False
        config.APEX_AMP = False
        config.NATIVE_AMP = False
    if args.no_compile:
        config.NO_COMPILE = True
    if args.output:
        config.OUTPUT = args.output
    if args.model_name:
        config.MODEL.NAME = args.model_name
    if args.load_test_dir:
        config.LOAD_TEST_DIR = args.load_test_dir
    if args.epochs:
        config.TRAIN.EPOCHS = args.epochs
    if args.local_rank:
        config.LOCAL_RANK = args.local_rank
    if args.pretrained_backbone:
        config.DATA.PRETRAINED_DIR = args.pretrained_backbone
    if args.train_mode:
        config.TRAIN_MODE = args.train_mode
    if args.ema:
        config.MODEL_EMA = args.ema
    if args.pin_memory:
        config.DATA.PIN_MEMORY = args.pin_memory

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.DISTRIBUTED = int(os.environ['WORLD_SIZE']) > 1
        config.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.PROJECT_NAME)
    config.freeze()

def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    if not args=='' and args is not None:
        update_config(config, args)

    return config

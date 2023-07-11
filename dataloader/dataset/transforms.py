import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from .utils import GaussianBlur,Solarization
import albumentations as A
from albumentations.pytorch import ToTensorV2
from timm.data.transforms import str_to_interp_mode
class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask

class DINOFGIA(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number,img_size,multi_view='strong_none'):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            GaussianBlur(0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            # Solarization(0.2),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            GaussianBlur(0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

        self.multi_view = multi_view

    def __call__(self, image):
        crops = []

        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class MULTIVIEWFGIA(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number,img_size,multi_view='strong_none'):

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.global_strong = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            GaussianBlur(1),
            Solarization(0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
            # Solarization(0.2),
            normalize,
        ])
        self.global_weak = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            GaussianBlur(0.1),
            transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.1),
            normalize,
        ])
        self.global_none = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            normalize,
        ])

        self.multi_view = multi_view

    def __call__(self, image):
        crops = []
        _multi_view = self.multi_view.split('_')
        for _view in _multi_view:
            if _view == 'strong':
                crops.append(self.global_strong(image))
            elif _view == 'weak':
                crops.append(self.global_weak(image))
            elif _view == 'none':
                crops.append(self.global_none(image))

        return crops

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number,img_size):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops

class SimSiamTransform():
    def __init__(self, image_size, mean_std=((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))):
        image_size = 224 if image_size is None else image_size # by default simsiam use image size 224
        p_blur = 0.5 if image_size > 32 else 0 # exclude cifar
        # the paper didn't specify this, feel free to change this value
        # I use the setting from simclr which is 50% chance applying the gaussian blur
        # the 32 is prepared for cifar training where they disabled gaussian blur
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size//20*2+1, sigma=(0.1, 2.0))], p=p_blur),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])
    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2 

class PicTMultiViewTransform():
    def __init__(self, is_train,type='strong_none',image_size=(224,224), mean_std=((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))):
        self.image_size = image_size
        self.mean_std = mean_std
        transform_strong = A.Compose([
                            A.RandomBrightnessContrast(p=0.7),
                            A.HorizontalFlip(p=0.7),
                            A.VerticalFlip(p=0.7),
                            A.ShiftScaleRotate(rotate_limit=15.0, p=0.7),
                            A.OneOf([
                                A.Emboss(p=1),
                                A.Sharpen(p=1),
                                A.Blur(p=1)
                                    ], p=0.7),
                            ])
                            
        transform_weak = A.Compose([
                            # A.Resize(height=image_size[0],width=image_size[1]),
                            A.RandomBrightnessContrast(p=0.5),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.5),
                            A.ShiftScaleRotate(rotate_limit=15.0, p=0.7),
                            # A.Normalize(mean_std[0], mean_std[1]),
                            # ToTensorV2(),
                            ])

        transform_no_aug = A.Compose([
                            ])
        if is_train:
            if type == 'strong_none':
                self.transform_s = transform_strong
                self.transform_t = transform_no_aug
            elif type == 'strong_weak':
                self.transform_s = transform_strong
                self.transform_t = transform_weak
            elif type == 'weak_none':
                self.transform_s = transform_weak
                self.transform_t = transform_no_aug
        else:
            self.transform_s,self.transform_t = transform_no_aug,transform_no_aug

    def __call__(self, x):
        x = transforms.Resize(size=self.image_size,interpolation=str_to_interp_mode('bicubic'))(img=x)

        x1 = self.transform_s(image = np.asarray(x))['image']

        x1 = transforms.ToTensor()(x1)
        x1 = transforms.Normalize(mean=self.mean_std[0],std=self.mean_std[1])(x1)

        x2 = self.transform_t(image = np.asarray(x))['image']

        x2 = transforms.ToTensor()(x2)
        x2 = transforms.Normalize(mean=self.mean_std[0],std=self.mean_std[1])(x2)

        return x1, x2 

class SimMIMTransform:
    def __init__(self, config):
        self.transform_img = transforms.Compose([
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.RandomResizedCrop(config.DATA.IMG_SIZE, scale=(0.67, 1.), ratio=(3. / 4., 4. / 3.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.AUG.NORM[0],
                            std=config.AUG.NORM[1],),
        ])
 
        if config.MODEL.TYPE == 'swin':
            model_patch_size=4
        elif config.MODEL.TYPE == 'vit':
            model_patch_size=config.MODEL.VIT.PATCH_SIZE
        else:
            raise NotImplementedError
        
        self.mask_generator = MaskGenerator(
            input_size=config.DATA.IMG_SIZE[0],
            mask_patch_size=config.MIM.MASK_PATCH_SIZE,
            model_patch_size=model_patch_size,
            mask_ratio=config.MIM.MASK_RATIO,
        )
    
    def __call__(self, img):
        img = self.transform_img(img)
        mask = self.mask_generator()
        
        return (img, mask)


# Target Transform, to binary target
# Default: 0 is the negative, and 1 is the positive
class ToBinTarget:
    def __init__(self, nor_index):
        self.nor_index = nor_index

    def __call__(self,target):
        return 0 if target == self.nor_index else 1


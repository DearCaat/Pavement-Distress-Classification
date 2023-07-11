import torch
from PIL import Image
import random
import os
import argparse
from torchvision import transforms
from timm.data.transforms import str_to_interp_mode
from config import _update_config_from_file
import numpy as np
import shutil
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models import build_model
from config import get_config

def parse_option():
    parser = argparse.ArgumentParser('WSPLIN training and evaluation script', add_help=False)
    parser.add_argument('--bin', action='store_true', help='Use thumb data')
    parser.add_argument('--num', type=int, default=10 ,help="batch size for single GPU")
    
    args, unparsed = parser.parse_known_args()
    return args

def model_init(cpt_path,cpt_b_path,cpt_ema_path=None,is_bin=False):
    
    config=get_config(None)
    if is_bin:
        _update_config_from_file(config, './configs/pict.yaml')
        _update_config_from_file(config, './configs/best/pict_bin_979.yaml')
    else:
        _update_config_from_file(config, './configs/pict.yaml')
        _update_config_from_file(config, './configs/best/pict_70.yaml')
    #_update_config_from_file(config, '/home/tangwenhao/pict_code/pictformer/configs/best/pict_bin_979.yaml')

    model = build_model(config)
    
    # /home/tangwenhao/output/pict_new_init/model/pict_swin_small_patch4_window7_224_his_best_model.pth
    # /home/tangwenhao/output/pict_bin_abl/model/pict_swin_small_patch4_window7_224clu_2_ema_best_model.pth
    cpt = torch.load(cpt_path, map_location='cpu')
    model.load_state_dict(cpt['state_dict'], strict=False)

    model_ema = build_model(config)
    if cpt_ema_path is not None:
        cpt_ema = torch.load(cpt_ema_path, map_location='cpu')
        model_ema.load_state_dict(cpt_ema['state_dict'], strict=False)

    config=get_config(None)
    if is_bin:
        _update_config_from_file(config, './configs/baseline/pict_swin_small_bin.yaml')
    else:
        _update_config_from_file(config, './configs/baseline/pict_swin_small.yaml')

    model_b = build_model(config)
    cpt = torch.load(cpt_b_path, map_location='cpu')
    model_b.load_state_dict(cpt['state_dict'], strict=False)
    _ = model_b.eval()

    return model,model_b,model_ema

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def main():
    args = parse_option()
    is_bin = args.bin if args.bin else False

    # cementation_fissures crack longitudinal_crack loose massive_crack mending normal transverse_crack
    if is_bin:
        class_map = ['cementation_fissures','crack', 'longitudinal_crack', 'loose', 'massive_crack', 'mending', 'normal', 'transverse_crack']
        cpt = '/home/tangwenhao/output/pict_bin_abl/model/pict_swin_small_patch4_window7_224clu_2_best_model.pth'
        cpt_b = '/home/tangwenhao/output/pict_swin_small_bin/model/swin_small_patch4_window7_224_best_model.pth'
        cpt_ema = '/home/tangwenhao/output/pict_bin_abl/model/pict_swin_small_patch4_window7_224clu_2_ema_best_model.pth'
        output = './output/heatmap/bin/'
    else:
        class_map = ['cementation_fissures','crack', 'longitudinal_crack', 'loose', 'massive_crack', 'mending', 'normal', 'transverse_crack']
        cpt = '/home/tangwenhao/output/pict_new_init/model/pict_swin_small_patch4_window7_224_his_best_model.pth'
        cpt_b = '/home/tangwenhao/output/swin/model/swin_small_patch4_window7_224_best_model.pth'
        cpt_ema = '/home/tangwenhao/output/pict_new_init/model/pict_swin_small_patch4_window7_224_his_ema_best_model.pth'
        output = './output/heatmap/mul/'

    if os.path.exists(output):
        shutil.rmtree(output)

    root_dir = '/home/tangwenhao/data/cqu_bpdd/test'

    if not os.path.exists(output):
            os.mkdir(output)
    num_imgs_per_class = args.num
    
    model,model_b,model_ema = model_init(cpt,cpt_b,cpt_ema,is_bin=is_bin)

    target_layers = [model.instance_feature_extractor.layers[-1].blocks[-1].norm1]
    target_layers_ema = [model_ema.instance_feature_extractor.layers[-1].blocks[-1].norm1]
    target_layers_b = [model_b.layers[-1].blocks[-1].norm1]
    cam_b = GradCAM(model=model_b, target_layers=target_layers_b, reshape_transform=reshape_transform)
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    cam_ema = GradCAM(model=model_ema, target_layers=target_layers_ema, reshape_transform=reshape_transform)

    for idx,class_name in enumerate(class_map):
        
        _dir = os.path.join(root_dir,class_name)
        _output = os.path.join(output,class_name)
        if not os.path.exists(_output):
            os.mkdir(_output)
        for root,dirs,files in os.walk(_dir):
            num_pic = len(files)
        for i in range(num_imgs_per_class):
            _file = random.randint(0, num_pic-1)
            img = Image.open(os.path.join(_dir,str(_file)+'.jpg')).convert('RGB')

            imgs = transforms.Resize(size=(224,224),interpolation=str_to_interp_mode('bicubic'))(img=img)
            imgs = transforms.ToTensor()(imgs)
            imgs = transforms.Normalize(mean=torch.tensor((0.485, 0.456, 0.406)),std=torch.tensor((0.229, 0.224, 0.225)))(imgs)
            imgs = imgs.unsqueeze(0)
            if is_bin:
                _idx = int(class_name != 'normal')
                target_category = [ClassifierOutputTarget(_idx)]
            else:
                target_category = [ClassifierOutputTarget(idx)]
            grayscale_cam = cam(input_tensor=imgs,
                                targets=target_category,)
            grayscale_cam_ema = cam_ema(input_tensor=imgs,
                                targets=target_category,)
            grayscale_cam_b = cam_b(input_tensor=imgs,
                    targets=target_category,)

            rgb_img = cv2.resize(np.array(img), (224, 224))
            rgb_img = np.float32(rgb_img) / 255

            grayscale_cam_b = grayscale_cam_b[0, :]
            cam_image_b = show_cam_on_image(rgb_img, grayscale_cam_b,use_rgb=True)
            grayscale_cam = grayscale_cam[0, :]
            cam_image = show_cam_on_image(rgb_img, grayscale_cam,use_rgb=True)
            grayscale_cam_ema = grayscale_cam_ema[0, :]
            cam_image_ema = show_cam_on_image(rgb_img, grayscale_cam_ema,use_rgb=True)

            save_path = os.path.join(_output,'baseline_'+str(i)+'.jpg')

            Image.fromarray(cam_image_b).save(save_path,quality=95)
            img.save(save_path.replace('baseline','ori'),quality=95)
            Image.fromarray(cam_image).save(save_path.replace('baseline','pict'),quality=95)
            Image.fromarray(cam_image_ema).save(save_path.replace('baseline','pict_ema'),quality=95)

            # with open('./output/heatmap/cat.txt','a',encoding='utf-8') as f:
            #     text = '\n'+save_path[26:-4]+' '+ class_name
            #     f.write(text)
            # f.close()

if __name__ == "__main__":

    main()
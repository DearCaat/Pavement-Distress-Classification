from PIL import Image
import numpy as np
# extract patches from 3 size of img, 1200x900, 600x600, 300x300
# total 12 + 6 + 1 patches
class PatchExtractor:
    def __init__(self, img, patch_size, stride,noIP=False):
        if type(img) == np.ndarray:
            img = Image.fromarray(img)
        if type(stride) not in (tuple,list):
            stride = (stride,stride)
        # 3 layers stride
        stride = np.array(stride).squeeze()
        if len(stride.shape) == 1:
            stride = np.expand_dims(stride,0).repeat(3,axis=0)
        self.img0 = img
        if noIP:
            self.img_list = [self.img0]
        else:
            self.img1 = img.resize((600, 600), Image.BILINEAR)
            self.img2 = img.resize((300, 300), Image.BILINEAR)
            self.img_list = [self.img0, self.img1, self.img2]

        self.size = patch_size
        self.stride = stride

    def extract_patches(self):

        patches = []
        for im,stride in zip(self.img_list,self.stride):
            wp, hp = self.shape(im,stride)
            temp = [self.extract_patch(im, (w, h),stride) for h in range(hp) for w in range(wp)]
            patches.extend(temp)
        return patches

    def extract_patch(self, img, patch,stride):


        return img.crop((
            patch[0] * stride[0],  # left
            patch[1] * stride[1],  # up
            patch[0] * stride[0] + self.size,  # right
            patch[1] * stride[1] + self.size  # down
        ))

    def shape(self, img,stride):
        wp = int((img.width - self.size) / stride[1] + 1)
        hp = int((img.height - self.size) / stride[0] + 1)
        return wp, hp



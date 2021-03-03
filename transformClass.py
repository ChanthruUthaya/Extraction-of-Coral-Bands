import torchvision.transforms.functional as TF
import random
from torchvision import transforms
from PIL import Image

class Rotation:

    def __init__(self, angle_range:tuple):
        self.l = angle_range[0] *100
        self.r = angle_range[1] *100

    def __call__(self, image, label):
        angle = random.randrange(self.l, self.r)/100
        return (TF.rotate(image, angle), TF.rotate(label, angle)) 


class Flip:

    def __init__(self, h_flip, v_flip):
        self.h_flip_prob = h_flip
        self.v_flip_prob = v_flip
    
    def __call__(self, image, label):

        image, label = (TF.hflip(image),TF.hflip(label)) if random.random() < self.h_flip_prob else (image,label)
        image, label = (TF.vflip(image),TF.vflip(label)) if random.random() < self.v_flip_prob else (image,label)

        return image, label

class AdjustBrightness:

    def __init__(self, brightness_range):
        self.l = brightness_range[0]
        self.r = brightness_range[1]

    def __call__(self, image, label):
        val = random.uniform(self.l,self.r)
        return (TF.adjust_brightness(image, val), label)

class Affine:
    def __init__(self, translate:tuple, shear_range:tuple, scale:float, angle:tuple):
        self.translate = translate
        self.shear_range = shear_range
        self.scale = scale
        self.angle = angle
    def __call__(self, image, label):
        scale_val = random.uniform(1-self.scale, 1+self.scale)
        shear_val = random.uniform(self.shear_range[0], self.shear_range[1])
        angle = random.uniform(self.angle[0], self.angle[1])
        h_trans_val = random.uniform(self.translate[0][0], self.translate[0][1])
        v_trans_val = random.uniform(self.translate[1][0], self.translate[1][1])

        image, label = (TF.affine(image,angle = angle, translate = (h_trans_val, v_trans_val) ,scale = scale_val, shear = shear_val), TF.affine(label, angle = angle, translate = (h_trans_val, v_trans_val), scale = scale_val, shear = shear_val)) 
       

        return image, label

class Transform:
    def __init__(self, flip:Flip, brightness:AdjustBrightness, affine:Affine):
        self.flip = flip
        self.brightness = brightness
        self.affine = affine
    
    def __call__(self, image, label):
        image, label = self.flip(image, label)
        #image, label = self.brightness(image, label)
        image, label = self.affine(image, label)

        return image, label


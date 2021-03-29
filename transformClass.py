import torchvision.transforms.functional as TF
import random
from torchvision import transforms
import PIL
from PIL import ImageEnhance
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

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
    
    def __call__(self, image, label, h_flip, v_flip):
        image, label = (TF.hflip(image),TF.hflip(label)) if h_flip else (image,label)
        image, label = (TF.vflip(image),TF.vflip(label)) if v_flip else (image,label)

        return image, label

class AdjustBrightness:

    def __init__(self, brightness_range):
        self.l = brightness_range[0]
        self.r = brightness_range[1]
        self.range = brightness_range
    
    @staticmethod
    def array_to_img(x, data_format='channels_last', scale=True, dtype=np.float32):
        """Converts a 3D Numpy array to a PIL Image instance.
        # Arguments
            x: Input Numpy array.
            data_format: Image data format, either "channels_first" or "channels_last".
                Default: "channels_last".
            scale: Whether to rescale the image such that minimum and maximum values
                are 0 and 255 respectively.
                Default: True.
            dtype: Dtype to use.
                Default: "float32".
        # Returns
            A PIL Image instance.
        # Raises
            ImportError: if PIL is not available.
            ValueError: if invalid `x` or `data_format` is passed.
        """
        x = np.asarray(x, dtype=dtype)
        print(np.max(x), x.shape)
        if x.ndim != 3:
            raise ValueError('Expected image array to have rank 3 (single image). '
                            'Got array with shape: %s' % (x.shape,))

        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Invalid data_format: %s' % data_format)

        # Original Numpy array x has format (height, width, channel)
        # or (channel, height, width)
        # but target PIL image has format (width, height, channel)
        if data_format == 'channels_first':
            x = x.transpose(1, 2, 0)
        if scale:
            x = x - np.min(x)
            x_max = np.max(x)
            if x_max != 0:
                x /= x_max
            x *= 255
        if x.shape[2] == 4:
            # RGBA
            return Image.fromarray(x.astype('uint8'), 'RGBA')
        elif x.shape[2] == 3:
            # RGB
            return Image.fromarray(x.astype('uint8'), 'RGB')
        elif x.shape[2] == 1:
            # grayscale
            if np.max(x) > 255:
                # 32-bit signed integer grayscale image. PIL mode "I"
                image = Image.fromarray(x[:, :, 0].astype('int32'), 'I')
                return image
            return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
        else:
            raise ValueError('Unsupported channel number: %s' % (x.shape[2],))

    @staticmethod
    def img_to_array(img, data_format='channels_last', dtype=np.uint16):
        """Converts a PIL Image instance to a Numpy array.
        # Arguments
            img: PIL Image instance.
            data_format: Image data format,
                either "channels_first" or "channels_last".
            dtype: Dtype to use for the returned array.
        # Returns
            A 3D Numpy array.
        # Raises
            ValueError: if invalid `img` or `data_format` is passed.
        """
        if data_format not in {'channels_first', 'channels_last'}:
            raise ValueError('Unknown data_format: %s' % data_format)
        # Numpy array x has format (height, width, channel)
        # or (channel, height, width)
        # but original PIL image has format (width, height, channel)
        x = np.asarray(img, dtype=dtype)
        if len(x.shape) == 3:
            if data_format == 'channels_first':
                x = x.transpose(2, 0, 1)
        elif len(x.shape) == 2:
            if data_format == 'channels_first':
                x = x.reshape((1, x.shape[0], x.shape[1]))
            else:
                x = x.reshape((x.shape[0], x.shape[1], 1))
        else:
            raise ValueError('Unsupported image shape: %s' % (x.shape,))
        return x
    
    @staticmethod
    def apply_brightness_shift(x, brightness, scale=True):
        """Performs a brightness shift.
        # Arguments
            x: Input tensor. Must be 3D.
            brightness: Float. The new brightness value.
            scale: Whether to rescale the image such that minimum and maximum values
                are 0 and 255 respectively.
                Default: True.
        # Returns
            Numpy image tensor.
        # Raises
            ImportError: if PIL is not available.
        """
        if ImageEnhance is None:
            raise ImportError('Using brightness shifts requires PIL. '
                            'Install PIL or Pillow.')
        x_min, x_max = np.min(x), np.max(x)
        local_scale = (x_min < 0) or (x_max > 255)
        x = AdjustBrightness.array_to_img(x, scale=local_scale or scale)
        x = imgenhancer_Brightness = ImageEnhance.Brightness(x)
        x = imgenhancer_Brightness.enhance(brightness)
        x = AdjustBrightness.img_to_array(x)
        if not scale and local_scale:
            x = x / 255 * (x_max - x_min) + x_min
        return x

    @staticmethod
    def random_brightness(x, brightness_range, scale=True):
        """Performs a random brightness shift.
        # Arguments
            x: Input tensor. Must be 3D.
            brightness_range: Tuple of floats; brightness range.
            scale: Whether to rescale the image such that minimum and maximum values
                are 0 and 255 respectively.
                Default: True.
        # Returns
            Numpy image tensor.
        # Raises
            ValueError if `brightness_range` isn't a tuple.
        """
        if len(brightness_range) != 2:
            raise ValueError(
                '`brightness_range should be tuple or list of two floats. '
                'Received: %s' % (brightness_range,))

        u = np.random.uniform(brightness_range[0], brightness_range[1])
        return AdjustBrightness.apply_brightness_shift(x, u, scale)

    def __call__(self, image, label):
        image = np.expand_dims(np.array(image).astype(np.uint16), axis=2)
        #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        #print(np.max(image))
       # print(image.shape)
       # image = cv2.convertScaleAbs(np.array(image).astype(np.uint16))
       # val = random.uniform(self.l,self.r)
        image = tf.keras.preprocessing.image.random_brightness(image, self.range)
        image = image.squeeze()
        #print(np.max(image))
        #image = np.clip(image*val, 0.0, 255.0)
        return image, np.array(label).astype(np.float32)

class Affine:
    def __init__(self, translate:tuple, shear_range:tuple, scale:float, angle:tuple):
        self.translate = translate
        self.shear_range = shear_range
        self.scale = scale
        self.angle = angle
    def __call__(self, image, label, scale_val, shear_val, angle, h_trans_val, v_trans_val):
        image, label = (TF.affine(image,
                                    angle = angle, 
                                    translate = (h_trans_val, v_trans_val),
                                    scale = scale_val, shear = shear_val), 
                        TF.affine(label, 
                                    angle = angle, 
                                    translate = (h_trans_val, v_trans_val), 
                                    scale = scale_val, 
                                    shear = shear_val)) 
       

        return image, label

class Transform:
    def __init__(self, flip:Flip, brightness:AdjustBrightness, affine:Affine, mode='2D'):
        self.flip = flip
        self.brightness = brightness
        self.affine = affine
        self.mode = mode

    def gen_flip_args(self):
        h_flip = True if random.random() < self.flip.h_flip_prob else False
        v_flip = True if random.random() < self.flip.v_flip_prob else False
        return h_flip, v_flip

    
    def gen_affine_args(self):
        scale_val = random.uniform(1-self.affine.scale, 1+self.affine.scale)
        shear_val = random.uniform(self.affine.shear_range[0], self.affine.shear_range[1])
        angle = random.uniform(self.affine.angle[0], self.affine.angle[1])
        h_trans_val = random.uniform(self.affine.translate[0][0], self.affine.translate[0][1])
        v_trans_val = random.uniform(self.affine.translate[1][0], self.affine.translate[1][1])

        return scale_val, shear_val, angle, h_trans_val, v_trans_val
    
    def __call__(self, image, label):

        if(self.mode == '2D'):
            h_flip, v_flip = self.gen_flip_args()
            scale_val, shear_val, angle, h_trans_val, v_trans_val = self.gen_affine_args()

            image, label = self.flip(image, label, h_flip, v_flip)
            image, label = self.affine(image, label, scale_val, shear_val, angle, h_trans_val, v_trans_val)
            image, label = self.brightness(image, label)

            return image, label
        
        elif(self.mode =='3D'):
            h_flip, v_flip = self.gen_flip_args()
            scale_val, shear_val, angle, h_trans_val, v_trans_val = self.gen_affine_args()
            res_img = []
            res_label = []
            for img, img_label in zip(image, label):
                if(str(type(img).__module__) == 'PIL.TiffImagePlugin'):
                    img, img_label = self.flip(img, img_label, h_flip, v_flip)
                    img, img_label = self.affine(img, img_label, scale_val, shear_val, angle, h_trans_val, v_trans_val)
                    img, img_label = self.brightness(img, img_label)

                res_img.append(img)
                res_label.append(img_label)

            return res_img, res_label


class TransformNew:
    def __init__(self, flip:Flip, brightness:AdjustBrightness, affine:Affine, mode='2D'):
        self.flip = flip
        self.brightness = brightness
        self.affine = affine
        self.mode = mode

    def gen_flip_args(self):
        h_flip = True if random.random() < self.flip.h_flip_prob else False
        v_flip = True if random.random() < self.flip.v_flip_prob else False
        return h_flip, v_flip

    
    def gen_affine_args(self):
        scale_val = random.uniform(1-self.affine.scale, 1+self.affine.scale)
        shear_val = random.uniform(self.affine.shear_range[0], self.affine.shear_range[1])
        angle = random.uniform(self.affine.angle[0], self.affine.angle[1])
        h_trans_val = random.uniform(self.affine.translate[0][0], self.affine.translate[0][1])
        v_trans_val = random.uniform(self.affine.translate[1][0], self.affine.translate[1][1])

        return scale_val, shear_val, angle, h_trans_val, v_trans_val
    
    def __call__(self, image, label):

        if(self.mode == '2D'):
            h_flip, v_flip = self.gen_flip_args()
            scale_val, shear_val, angle, h_trans_val, v_trans_val = self.gen_affine_args()

            image, label = self.flip(image, label, h_flip, v_flip)
            image, label = self.affine(image, label, scale_val, shear_val, angle, h_trans_val, v_trans_val)
            image, label = self.brightness(image, label)

            return image, label
        
        elif(self.mode =='3D'):
            h_flip, v_flip = self.gen_flip_args()
            scale_val, shear_val, angle, h_trans_val, v_trans_val = self.gen_affine_args()
            res_img = []
            res_label = []
            for img, img_label in zip(image, label):
                # if(str(type(img).__module__) == 'PIL.TiffImagePlugin'):
                img, img_label = self.flip(img, img_label, h_flip, v_flip)
                img, img_label = self.affine(img, img_label, scale_val, shear_val, angle, h_trans_val, v_trans_val)
                img, img_label = self.brightness(img, img_label)

                res_img.append(img)
                res_label.append(img_label)

            return res_img, res_label




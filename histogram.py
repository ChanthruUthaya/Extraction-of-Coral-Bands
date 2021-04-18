import os
import cv2 as cv
from glob import glob
import math
import re
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


# dir = os.readlink('scratch')

# path = dir + "/images"
# label_path = dir + "/labels"

# save_dir = dir + "/data/train/images/"
# save_label = dir + "/data/train/labels/"

dir = "D:/2D-remake/3ddata/chunk1"
# dir = "D:/2D-remake/data/train/"
#dir = os.readlink('scratch')


image_path = dir + "/sub/images"
#label_path = dir+ "/crop_label_resize/"
label_path = dir+ "/sub/labels"

save_test_dir = label_path + "/thresh/"
save_dir = dir + "/sub/"

# save_dir = os.readlink('scratch') + "/train"
# save_test_dir = os.readlink('scratch') + "/test"

def adjust_data(label):
        label = (label / 255.0)
        label[label > 0.5] = 1
        label[label <= 0.5] = 0

        return label*255.0

def adjust_image(image):
        
        return image

labels = [os.path.splitext(file)[0] for file in os.listdir(label_path)]

for i, label in enumerate([labels[0]]):

        img_file = glob(label_path + '/'+ label + '.*')[0].replace('\\', '/')

        image_pil = Image.open(img_file)
        print(image_pil.getextrema())
        image_pil.show()

        image = cv.imread(img_file, 0)
        print(image.shape)
        print(np.max(image))
        image = adjust_data(image)
        print(image.shape)
        print(np.unique(image))

        cv.imwrite(save_test_dir + f"/{i}-label.png", image)

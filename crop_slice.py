import os
import cv2 as cv
from glob import glob
import math


# dir = os.readlink('scratch')

# path = dir + "/images"
# label_path = dir + "/labels"

# save_dir = dir + "/data/train/images/"
# save_label = dir + "/data/train/labels/"

path = "D:/2D-remake/3ddata/chunk1/images/"
label_path = "D:/2D-remake/3ddata/chunk1/labels/"

save_dir = "D:/2D-remake/3ddata/chunk1/crop_resize/"
save_label = "D:/2D-remake/3ddata/chunk1/crop_label_resize/"

print(save_dir)
print(save_label)

label_prefix = "_labels"

size = 512
stride = 512

scale_percent = 0.5

images = [os.path.splitext(file)[0] for file in os.listdir(path)]
labels = [os.path.splitext(file)[0] for file in os.listdir(label_path)]

print(images)
#print(labels)

for (i,image) in enumerate(images[:2]):
    print(image)
    number = int(image.split('_')[-1])
    name = '_'.join(image.split('_')[:-1])
    img_file = glob(path + '/'+ image + '.*')[0].replace('\\', '/')
    label_file = glob(label_path + '/'+ image + label_prefix + '.*')[0].replace('\\', '/')
    img = cv.imread(img_file, -1)
    label = cv.imread(label_file, 0)
    crop_image = img[100:img.shape[0]-100,200:img.shape[1]-100]
    x, y = img.shape
    dim = (int(x*scale_percent), int(y*scale_percent))
    print(dim)
    crop_label = label[100:label.shape[0]-100,200:label.shape[1]-100]
    resize_image = cv.resize(img, dim, interpolation=cv.INTER_CUBIC)
    resize_label = cv.resize(label, dim, interpolation=cv.INTER_CUBIC)
    cv.imwrite(save_dir+f"{image}.tif", resize_image)
    cv.imwrite(save_label+f"{image}{label_prefix}.png", resize_label)

import os
import cv2 as cv
import numpy as np
from glob import glob
import math
from PIL import Image


#dir = os.readlink('scratch')
dir = "D:/2D-remake/3ddata/chunk1/"

# path = dir + "/images"
# label_path = dir + "/labels"

# save_dir = dir + "/data/train/images/"
# save_label = dir + "/data/train/labels/"

path = dir + "/images"
label_path = dir + "/labels"

save_dir = dir + "/crop_resize/"
save_label = dir + "/crop_label_resize/"

print(save_dir)
print(save_label)

label_prefix = "_labels"

size = 512
stride = 512

scale_percent = 0.5
brightness_factor = 1.2

images = [os.path.splitext(file)[0] for file in os.listdir(path)]
labels = [os.path.splitext(file)[0] for file in os.listdir(label_path)]

#print(images)
#print(labels)

for (i,image) in enumerate(images):
    print(image)
    number = int(image.split('_')[-1])
    name = '_'.join(image.split('_')[:-1])
    img_file = glob(path + '/'+ image + '.*')[0].replace('\\', '/')
    label_file = glob(label_path + '/'+ image + label_prefix + '.*')[0].replace('\\', '/')
    img = cv.imread(img_file, -1)
    label = cv.imread(label_file,0)
    print(label.shape)
    x, y = img.shape
    dim = (int(x*scale_percent), int(y*scale_percent))

    # ig1 = Image.fromarray(label).convert("LA")
    # print(ig1.size)
    # ig1.show()

    # print("before resize:")
    # print(np.max(label), np.max(img))
    # print(np.mean(label), np.mean(img))
    resize_image = cv.resize(img, dim, interpolation=cv.INTER_CUBIC)
    resize_label = cv.resize(label, dim, interpolation=cv.INTER_CUBIC)*brightness_factor
    crop_image = resize_image[40:resize_image.shape[0]-70,120:resize_image.shape[1]-60]
    crop_label = resize_label[40:resize_label.shape[0]-70,120:resize_label.shape[1]-60]
    # print("after resize:")
    # print(np.max(resize_label), np.max(resize_image))
    # print(np.mean(resize_label), np.mean(resize_image))
    cv.imwrite(save_dir+f"{image}.png", crop_image)
    cv.imwrite(save_label+f"{image}{label_prefix}.png", crop_label)


# resize_image = cv.resize(img, dim, interpolation=cv.INTER_LINEAR)
#     resize_label = cv.resize(label, dim, interpolation=cv.INTER_LINEAR)
#     crop_image =resize_image[50:resize_image.shape[0]-80,120:resize_image.shape[1]-70]
#     crop_label = resize_label[50:resize_label.shape[0]-80,120:resize_label.shape[1]-70]

#     print("before ", np.mean(crop_label), np.max(crop_label))

#     crop_image =crop_image*brightness_factor
#     crop_label = crop_label*brightness_factor
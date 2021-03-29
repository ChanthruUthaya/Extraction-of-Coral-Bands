import os
import cv2 as cv
from glob import glob
import math


# dir = os.readlink('scratch')

# path = dir + "/images"
# label_path = dir + "/labels"

# save_dir = dir + "/data/train/images/"
# save_label = dir + "/data/train/labels/"

path = "D:/2D-remake/3ddata/chunk1/crop/"
label_path = "D:/2D-remake/3ddata/chunk1/crop_label/"

save_dir = "D:/2D-remake/3ddata/chunk1/train/cuts_images/"
save_label = "D:/2D-remake/3ddata/chunk1/train/cuts_labels/"

print(save_dir)
print(save_label)

label_prefix = "_labels"

size = 256
stride = 20

images = [os.path.splitext(file)[0] for file in os.listdir(path)]
labels = [os.path.splitext(file)[0] for file in os.listdir(label_path)]

print(images)
#print(labels)

for (i,image) in enumerate([images[0]]):
    number = int(image.split('_')[-1])
    name = '_'.join(image.split('_')[:-1])
    img_file = glob(path + '/'+ image + '.*')[0].replace('\\', '/')
    label_file = glob(label_path + '/'+ image + label_prefix + '.*')[0].replace('\\', '/')
    image = cv.imread(img_file, -1)
    label = cv.imread(label_file, 0)
    bottom_x, bottom_y = image.shape
    print(bottom_x, bottom_y)
    number_x = (bottom_x - 256)/stride
    number_y = (bottom_y - 256)/stride
    print(f'{i}/{len(images)}')
    print(f"{math.floor(number_x)*math.floor(number_y)} images created")
    for x in range(math.floor(number_x)):
        for y in range(math.floor(number_y)):
            x1 = x * stride
            y1 = y * stride
            cropped_image = image[y1:y1+size, x1:x1+size]
            cropped_label = label[y1:y1+size, x1:x1+size]
            cv.imwrite(save_dir + f"{name}-{x}-{y}-{number}-0.tif", cropped_image)
            cv.imwrite(save_label + f"{name}-{x}-{y}-{number}-0-label.png", cropped_label)
            cropped_image = cv.rotate(cropped_image, cv.ROTATE_90_CLOCKWISE)
            cropped_label = cv.rotate(cropped_label, cv.ROTATE_90_CLOCKWISE)
            cv.imwrite(save_dir + f"{name}-{x}-{y}-{number}-1.tif", cropped_image)
            cv.imwrite(save_label + f"{name}-{x}-{y}-{number}-1-label.png", cropped_label)

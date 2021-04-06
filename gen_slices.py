import os
import cv2 as cv
from glob import glob
import math
import re


# dir = os.readlink('scratch')

# path = dir + "/images"
# label_path = dir + "/labels"

# save_dir = dir + "/data/train/images/"
# save_label = dir + "/data/train/labels/"

image_path = "D:/2D-remake/3ddata/chunk1/crop_resize/"
label_path = "D:/2D-remake/3ddata/chunk1/crop_label_resize/"

save_dir = "D:/2D-remake/3ddata/chunk1/train/"
save_label = "D:/2D-remake/3ddata/chunk1/train/"

print(save_dir)
print(save_label)

label_prefix = "_labels"

size = 256
stride = 100

max_dim = 22
excluded = []

def not_in(string):
    excluded = ['01490', '01520']
    if string not in excluded:
        return True
    else:
        return False

images = [os.path.splitext(file)[0] for file in os.listdir(image_path)]
test = [os.path.splitext(file)[0] for file in os.listdir(image_path) if not_in(os.path.splitext(file)[0].split("_")[-1])]
labels = [os.path.splitext(file)[0] for file in os.listdir(label_path)]

print(test)

file_read = open('exclude.txt', 'r')
lines = file_read.readlines()
for line in lines:
    line = line.replace('\n', "").strip()
    first, second = line.split("-")
    r_1 = tuple(map(int, first.split(",")))
    r_2 = tuple(map(int, second.split(",")))
    #print(r_1, r_2)
    mult_r_1 = r_1[0]
    mult_r_2 = r_2[0]
    x = list(range(mult_r_1*max_dim+r_1[1], mult_r_2*max_dim+r_2[1]+1))
    out = [(i//max_dim, i%max_dim) for i in x]
    excluded = excluded + out

# print(images)
#print(labels)

for (i,image_name) in enumerate([images[0]]):
    print(image_name)
    number = int(image_name.split('_')[-1])
    name = '_'.join(image_name.split('_')[:-1])
    #print(name, number)
    dir_to_save = f"{save_dir}{name}_{number}"
    image_dir = f"{save_dir}/new_images"
    label_dir = f"{save_dir}/new_labels"
    # if not os.path.exists(dir_to_save):
    #     os.makedirs(dir_to_save)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    img_file = glob(image_path + '/'+ image_name + '.*')[0].replace('\\', '/')
    label_file = glob(label_path + '/'+ image_name + label_prefix + '.*')[0].replace('\\', '/')
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
            if((x,y) not in excluded):
            #print("saving in: ", image_dir + f"/{name}-{x}-{y}-{number}-0.tif")
                cv.imwrite(image_dir + f"/{name}-{x}-{y}-{number}-0.tif", cropped_image)
                cv.imwrite(label_dir + f"/{name}-{x}-{y}-{number}-0-label.png", cropped_label)
                cropped_image = cv.rotate(cropped_image, cv.ROTATE_90_CLOCKWISE)
                cropped_label = cv.rotate(cropped_label, cv.ROTATE_90_CLOCKWISE)
                cv.imwrite(image_dir + f"/{name}-{x}-{y}-{number}-1.tif", cropped_image)
                cv.imwrite(label_dir + f"/{name}-{x}-{y}-{number}-1-label.png", cropped_label)

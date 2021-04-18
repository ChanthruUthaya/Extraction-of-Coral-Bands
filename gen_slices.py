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

dir = "D:/2D-remake/3ddata/chunk1/val1619-1621"
#dir = os.readlink('scratch')


image_path = dir + "/crop_resize/"
label_path = dir+ "/crop_label_resize/"

save_test_dir = dir + "/test/"
save_dir = dir + "/train/"

# save_dir = os.readlink('scratch') + "/train"
# save_test_dir = os.readlink('scratch') + "/test"



print(save_dir)

label_prefix = "_labels"

size = 256
stride = 100

max_dim = 22
excluded = []

# def not_in(string):
#     excluded_layer = ['01490', '01520']
#     if string not in excluded_layer:
#         return True
#     else:
#         return False

images = [os.path.splitext(file)[0] for file in os.listdir(image_path)]
images_train = images
images_test = []
#mid = len(images)//2 -1
#images = [images[0],images[-1],images[mid]] 
#test = [os.path.splitext(file)[0] for file in os.listdir(image_path) if not_in(os.path.splitext(file)[0].split("_")[-1])]



print(images)


#exclude columns
# file_read = open('exclude.txt', 'r')
# lines = file_read.readlines()
# for line in lines:
#     line = line.replace('\n', "").strip()
#     first, second = line.split("-")
#     r_1 = tuple(map(int, first.split(",")))
#     r_2 = tuple(map(int, second.split(",")))
#     #print(r_1, r_2)
#     mult_r_1 = r_1[0]
#     mult_r_2 = r_2[0]
#     x = list(range(mult_r_1*max_dim+r_1[1], mult_r_2*max_dim+r_2[1]+1))
#     out = [(i//max_dim, i%max_dim) for i in x]
#     excluded = excluded + out

# print(images)
#print(labels)

for (i,image_name) in enumerate(images):
    print(image_name)
    number = int(image_name.split('_')[-1])
    name = '_'.join(image_name.split('_')[:-1])
    #print(name, number)
    dir_to_save = f"{save_dir}{name}_{number}"
    image_dir = f"{save_dir}/images"
    label_dir = f"{save_dir}/labels"
    test_dir = f"{save_test_dir}/images"
    label_test_dir = f"{save_test_dir}/labels"
    # if not os.path.exists(dir_to_save):
    #     os.makedirs(dir_to_save)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    if not os.path.exists(label_test_dir):
        os.makedirs(label_test_dir)

    img_file = glob(image_path + '/'+ image_name + '.*')[0].replace('\\', '/')
    label_file = glob(label_path + '/'+ image_name + label_prefix + '.*')[0].replace('\\', '/')
    image = cv.imread(img_file, -1)
    label = cv.imread(label_file,0)
    bottom_x, bottom_y = image.shape
    print(bottom_x, bottom_y)
    number_x = (bottom_x - 256)/stride
    number_y = (bottom_y - 256)/stride
    print(f'{i}/{len(images)}')
    print(f"{math.floor(number_x)*math.floor(number_y)} images created")
    #if(image_name in images_train + images_test):
    for x in range(math.floor(number_x)):
        for y in range(math.floor(number_y)):
            x1 = x * stride
            y1 = y * stride
            cropped_image = image[y1:y1+size, x1:x1+size]
            cropped_label = label[y1:y1+size, x1:x1+size]
            # if((x,y) not in excluded):
            #print("saving in: ", image_dir + f"/{name}-{x}-{y}-{number}-0.tif")
            if(image_name in images_train):
                cv.imwrite(image_dir + f"/{name}-{x}-{y}-{number}-0.png", cropped_image)
                cv.imwrite(label_dir + f"/{name}-{x}-{y}-{number}-0-label.png", cropped_label)
                cropped_image_train = cv.rotate(cropped_image, cv.ROTATE_90_CLOCKWISE)
                cropped_label_train = cv.rotate(cropped_label, cv.ROTATE_90_CLOCKWISE)
                cv.imwrite(image_dir + f"/{name}-{x}-{y}-{number}-1.png", cropped_image_train)
                cv.imwrite(label_dir + f"/{name}-{x}-{y}-{number}-1-label.png", cropped_label_train)
            if(image_name in images_test): #save test
                cv.imwrite(test_dir + f"/{name}-{x}-{y}-{number}-0.png", cropped_image)
                cv.imwrite(label_test_dir + f"/{name}-{x}-{y}-{number}-0-label.png", cropped_label)
                cropped_image_test = cv.rotate(cropped_image, cv.ROTATE_90_CLOCKWISE)
                cropped_label_test = cv.rotate(cropped_label, cv.ROTATE_90_CLOCKWISE)
                cv.imwrite(test_dir + f"/{name}-{x}-{y}-{number}-1.png", cropped_image_test)
                cv.imwrite(label_test_dir + f"/{name}-{x}-{y}-{number}-1-label.png", cropped_label_test)


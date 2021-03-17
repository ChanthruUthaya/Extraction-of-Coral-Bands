from glob import glob
import numpy as np
import os
import skimage.io as io
# import skimage.transform as trans
from skimage import img_as_ubyte, morphology
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
import random

import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
import cv2 as cv

# class CoralDataset(Dataset):
#     def __init__(self, img_dir, label_dir,augmentations, mode, label_suffix = '_label'):
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.label_suffix = label_suffix
#         self.augmentations = augmentations
#         self.mode = mode
#         self.ids = [os.path.splitext(file)[0] for file in os.listdir(img_dir)]
    
#     def __len__(self):
#         return len(self.ids)
    
#     @staticmethod
#     def adjust_data(image, label):
#         if np.max(image) > 1:
#             image = image / 255
#             label = label / 255
#             label[label > 0.5] = 1
#             label[label <= 0.5] = 0

#         return Image.fromarray(image), Image.fromarray(label)

#     def augment(self, img, label):
#         img, label = CoralDataset.adjust_data(img, label)
#         #img, label = Image.fromarray(img), Image.fromarray(label)
#         trans = transforms.Compose([
#                 transforms.ToTensor()
#                 ])
#         if(self.mode == 0):
#             img, label = self.augmentations(img, label)
#         return trans(img), trans(label)

#     def __str__(self):
#         return '[' + ' , '.join(self.ids) + ']'

#     def __getitem__(self, i):

#         indexes = [j for j in range(len(self.ids)) if j != i]
#         afteridx = random.choice(indexes)
#         indexes.remove(afteridx)
#         beforeidx = random.choice(indexes)

#         idx = [self.ids[beforeidx], self.ids[i], self.ids[afteridx]]

#         label_file = [glob(self.label_dir + '/' + j + self.label_suffix + '.*')[0] for j in idx]
#         img_file = [glob(self.img_dir + '/'+ j + '.*')[0] for j in idx]

#         img_0 = Image.open(img_file[1].replace('\\', '/'))
#         label_0 = Image.open(label_file[1].replace('\\', '/'))

#         labels = [np.array(Image.open(j.replace('\\', '/'))) for j in label_file]
#         images = [np.array(Image.open(j.replace('\\', '/'))) for j in img_file]

#         # zips = list(map(self.augment, images, labels))
#         # img, label, weights = torch.stack([i for i,_,_ in zips]), torch.stack([j for _, j,_ in zips]), torch.stack([k for _,_,k in zips])

#         # if i == 2:
#         #     img_0.show()
#         #     label_0.show()

#         img_0, label_0 = self.augment(np.array(img_0), np.array(label_0))

#         # if i == 2:
#         #     img_0.show()
#         #     label_0.show()


#         # return {
#         #     'image': img_0,
#         #     'label': label_0
#         # }
#         return {
#             'image': img_0,
#             'label': label_0,
#         }

class CoralDataset2D(Dataset):
    """2D Coral slices dataset."""

    def __init__(self, sample_dir, label_dir, transform=None):
        self.sample_dir = sample_dir
        self.label_dir = label_dir
        self.transform = transform

    def __len__(self):
        return len(glob(f"{self.sample_dir}/*.png"))

    def __getitem__(self, idx):
        f = sorted(glob(f"{self.sample_dir}/*.png"))[idx]
        name = os.path.abspath(f)
        sample = io.imread(name)
        sample = transforms.functional.to_pil_image(sample)

        f = sorted(glob(f"{self.label_dir}/*.png"))[idx]
        name = os.path.abspath(f)
        label = io.imread(name)
        threshold = label < 0.5
        label[threshold] = 0
        label = transforms.functional.to_pil_image(label)

        if self.transform:
            sample = self.transform(sample)
            label = self.transform(label)

        return sample, label


class CoralDataset(Dataset):
    def __init__(self, dir ,augmentations, mode, label_suffix = '_label', aug_dict=dict()):
        self.dir = dir
        self.img_dir = dir+"/image"
        self.label_dir = dir + "/label"
        self.label_suffix = label_suffix
        self.augmentations = augmentations
        self.mode = mode
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(self.img_dir)]
        self.aug_dict = aug_dict


        # for image, label in self.train_generator:
        #     print("adding")
        #     image, label = CoralDataset.adjust_data(image, label)
        #     self.data.append((image,label))

    def generator(self):
        image_datagen = ImageDataGenerator(**self.aug_dict)
        label_datagen = ImageDataGenerator(**self.aug_dict)

        seed = np.random.randint(0, 100)

    # The same seed argument is used when the image and label generators are
    # created to ensure that the same transformations are applied to both.
        image_generator = image_datagen.flow_from_directory(
            self.dir,
            classes=["image"],
            class_mode=None,
            color_mode="grayscale",
            target_size=(256,256),
            batch_size=2,
            save_to_dir=None,
            save_prefix="image",
            seed=seed
        )

        label_generator = label_datagen.flow_from_directory(
            self.dir,
            classes=["label"],
            class_mode=None,
            color_mode="grayscale",
            target_size=(256,256),
            batch_size=2,
            save_to_dir=None,
            save_prefix="label",
            seed=seed
        )

        # Zip the two generators into one.
        train_generator = zip(image_generator, label_generator)

        for image, label in train_generator:
            image, label = CoralDataset.adjust_data(image, label)
            image, label = torch.from_numpy(image), torch.from_numpy(label)
            yield image.view(2,-1,256,256), label.squeeze()

    
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def adjust_data(image, label):
        if np.max(image) > 1:
            image = image / 255
            label = label / 255
            label[label > 0.5] = 1
            label[label <= 0.5] = 0

        return image, label

    def augment(self, img, label):
        img, label = CoralDataset.adjust_data(img, label)
        img, label = Image.fromarray(img), Image.fromarray(label)
        trans = transforms.Compose([
                #transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                ])
        if(self.mode == 0):
            img, label = self.augmentations(img, label)
        return trans(img), trans(label)

    def __str__(self):
        return '[' + ' , '.join(self.ids) + ']'

    def __getitem__(self, i):

        indexes = [j for j in range(len(self.ids)) if j != i]
        afteridx = random.choice(indexes)
        indexes.remove(afteridx)
        beforeidx = random.choice(indexes)

        idx = [self.ids[beforeidx], self.ids[i], self.ids[afteridx]]

        #print(self.ids[i])

        # label_file = [glob(self.label_dir + '/' + j + self.label_suffix + '.*')[0] for j in idx]
        # img_file = [glob(self.img_dir + '/'+ j + '.*')[0] for j in idx]

        label_file = glob(self.label_dir + '/' + self.ids[i] + self.label_suffix + '.*')[0]
        img_file = glob(self.img_dir + '/'+ self.ids[i] + '.*')[0]

        img_0 = Image.open(img_file.replace('\\', '/'))
        label_0 = Image.open(label_file.replace('\\', '/'))


        #img_0.show()

        #print(str(img_file))


        # labels = [np.array(Image.open(j.replace('\\', '/'))) for j in label_file]
        # images = [np.array(Image.open(j.replace('\\', '/'))) for j in img_file]

        img_0, label_0 = self.augment(np.array(img_0), np.array(label_0))
        # img, label = self.data[i]
        # img, label - self.augment(img, label)



        # print("img size",img_0.shape)
        # print("label size", label_0.shape)

        return {
            'image': img_0,
            'label': label_0
        }

# def save_predictions(save_path, predictions):
#     for i, batch in enumerate(predictions):
#         for j, item in enumerate(batch):

#             image = item[0, :, :]
#            #
#            #  label_img = label[0, :, :]

#             output = img_as_ubyte(image)
#            # output_label = img_as_ubyte(label_img)

#             # Threshold the image using Otsu's method.
#             _, output = cv.threshold(output, 0, 255, cv.THRESH_OTSU)

#             # Replace all 255s with 1 in preparation for the skeletonization.
#             output[output == 255] = 1

#             # conf = gather_data(output, output_label)
#             # print(conf)
#             # cer, f_val = metrics(conf)
#             # print(f'CER val {cer}, F score {f_val}'



# # Skeletonize the thresholded predictions.
#             skel = morphology.skeletonize(output)
#             skel = skel.astype(int) * 255

# #Output the skeletonized prediction.
#             print("Saving prediction to out.png")

#             cv.imwrite(os.path.join(save_path, f"{i}{j}_predict.png"), np.array(skel))

def save_predictions(save_path, image, i):
    #image = image[0, :, :]
#
#  label_img = label[0, :, :]

    output = img_as_ubyte(image)
#  # output_label = img_as_ubyte(label_img)

#   # Threshold the image using Otsu's method.
    _, output = cv.threshold(output, 0, 255, cv.THRESH_OTSU)

#   # Replace all 255s with 1 in preparation for the skeletonization.
    output[output == 255] = 1

#   # Skeletonize the thresholded predictions.
    skel = morphology.skeletonize(output)
    skel = skel.astype(int) * 255

    #Output the skeletonized prediction.
    print("Saving prediction to out.png")

    print(output.shape)

    cv.imwrite(os.path.join(save_path, f"{i}_predict.png"), skel)

def save_pred(save_path, image, i):
        #image = image[0, :, :]
#
#  label_img = label[0, :, :]

    output = img_as_ubyte(image)
#  # output_label = img_as_ubyte(label_img)
    #Output the skeletonized prediction.
    print("Saving prediction to out.png")

    print(output.shape)

    cv.imwrite(os.path.join(save_path, f"{i}_predict.png"), output)



# train_dataset = datasets.ImageFolder(
#     traindir,
#     transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         normalize,
#     ]))
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
#     num_workers=args.workers, pin_memory=True, sampler=train_sampler)


# torchvision.transforms.functional.affine -- shear
# torchvision.transforms.RandomAffine -- translate (shift H + W)
# torchvision.transforms.RandomAffine -- zoom ()
#fill_mode = nearest
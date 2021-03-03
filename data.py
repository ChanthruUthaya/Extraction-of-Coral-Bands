from glob import glob
import numpy as np
import os
import skimage.io as io
# import skimage.transform as trans
from skimage import img_as_ubyte
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image
import random

class CoralDataset(Dataset):
    def __init__(self, img_dir, label_dir,augmentations, mode, label_suffix = '_label'):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.label_suffix = label_suffix
        self.augmentations = augmentations
        self.mode = mode
        self.ids = [os.path.splitext(file)[0] for file in os.listdir(img_dir)]
    
    def __len__(self):
        return len(self.ids)
    
    @staticmethod
    def adjust_data(image, label):
        if np.max(image) > 1:
            image = image / 255
            label = label / 255
            label[label > 0.5] = 1
            label[label <= 0.5] = 0
        return Image.fromarray(image), Image.fromarray(label)

    def augment(self, img, label):
        img, label = CoralDataset.adjust_data(img, label)
        trans = transforms.Compose([
                transforms.ToTensor()
                ])
        if(self.mode == 0): #for train and validation
            #img, label = self.augmentations(img, label)
            return trans(img), trans(label)
        return trans(img), trans(label)

    def __str__(self):
        return '[' + ' , '.join(self.ids) + ']'

    def __getitem__(self, i):

        indexes = [j for j in range(len(self.ids)) if j != i]
        afteridx = random.choice(indexes)
        indexes.remove(afteridx)
        beforeidx = random.choice(indexes)

        idx = [self.ids[beforeidx], self.ids[i], self.ids[afteridx]]

        label_file = [glob(self.label_dir + '/' + j + self.label_suffix + '.*')[0] for j in idx]
        img_file = [glob(self.img_dir + '/'+ j + '.*')[0] for j in idx]

        img_0 = Image.open(img_file[1].replace('\\', '/'))
        label_0 = Image.open(label_file[1].replace('\\', '/'))

        labels = [np.array(Image.open(j.replace('\\', '/'))) for j in label_file]
        images = [np.array(Image.open(j.replace('\\', '/'))) for j in img_file]

        zips = list(map(self.augment, images, labels))
        img, label = torch.stack([i for i,_ in zips]), torch.stack([j for _, j in zips])

        

        img_0, label_0 = self.augment(np.array(img_0), np.array(label_0))


        return {
            'image': img_0,
            'label': label_0
        }
        # return {
        #     'image': img,
        #     'label': label
        # }

def save_predictions(save_path, predictions):
    for i, batch in enumerate(predictions):
        for j, item in enumerate(batch):
            image = item[0, :, :]
            io.imsave(os.path.join(save_path, f"{i}{j}_predict.png"), img_as_ubyte(image))


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
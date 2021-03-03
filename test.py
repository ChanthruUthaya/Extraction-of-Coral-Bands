
import argparse
from data import *
from models import *
from tools import *
import os
import torch
import torch.backends.cudnn
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from multiprocessing import cpu_count
from pathlib import Path

import time


parser = argparse.ArgumentParser()
parser.add_argument("--resume-checkpoint", type=Path, default=Path("./checkpoint/checkpoint"))
parser.add_argument("--batch-size", default=2, type=int, help="Number of images within each mini-batch")
parser.add_argument("--dir", type=Path, default=Path("./data"))
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes used to load data.")
args = parser.parse_args()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    model = UNET(classes=2, height=256, width=256, channels=1)


    ### CHECKPOINT - load parameters, args, loss ###
    if args.resume_checkpoint != None and args.resume_checkpoint.exists():
        if torch.cuda.is_available():
            checkpoint = torch.load(args.resume_checkpoint)
        else:
            # if CPU is used
            checkpoint = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))

        print(f"Testing model {args.resume_checkpoint} that achieved {checkpoint['loss']} loss")

        model.load_state_dict(checkpoint['model'])

    images = str(args.dir) + "/test/image"
    labels = str(args.dir) + "/test/label"
    transform = [
        transforms.ToTensor()
        ]

    test_data = CoralDataset(img_dir=images, label_dir=labels, augmentations=transform, mode=1)

    test_loader = DataLoader(test_data, shuffle=False ,batch_size=args.batch_size, num_workers=args.worker_count, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()

   # preds = np.empty([0, 2304])
    total_loss = 0
    model.eval()
    preds = []

        # No need to track gradients for validation, we're not optimizing.
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch['image']
            labels = batch['label']
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(image)
            loss = criterion(logits, labels)
            print(f'[{i}/{len(test_loader)}] batch at loss: {loss.item()}')
            total_loss += loss.item()
            preds.append(logits.cpu().numpy())
           # preds = np.vstack((preds, logits.cpu().numpy()))

    average_loss = total_loss / len(test_loader)

    print(average_loss)
    print(len(preds))
    save_predictions("./predictions",preds)


if __name__ == '__main__':
    main(parser.parse_args())
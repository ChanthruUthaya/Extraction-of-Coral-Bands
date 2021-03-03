import torch
import numpy as np


class EarlyStopping:

    def __init__(self, delta, patience):
        self.delta = delta
        self.patience = patience
        self.previous = None
        self.diff = None
        self.best_Score = None

    def __call__(self, val_loss):
        if self.previous is None and self.diff is None:
            self.diff = val_loss
            self.previous = val_loss
        else:
            self.diff = self.previous - val_loss
            self.previous = val_loss

class ModelCheckpoint:

    def __init__(self, args):
        self.frequency = args.checkpoint_fq
        self.path = args.checkpoint_path
        self.epochs = args.epochs
        self.args = args
    
    def __call__(self, model, loss, epoch):
        ### CHECKPOINT - save parameters, args, accuracy ###
            #Save every args.checkpoint_frequency or if this is the last epoch
        if (epoch + 1) % self.frequency == 0 or (epoch + 1) == self.epochs:
            print(f"Saving model to {self.path}")
            torch.save({
                'args': self.args,
                'model': model.state_dict(),
                'loss': loss.item()
            }, self.path)


        



        


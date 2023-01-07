from abc import abstractmethod, ABC
import os

import torch

from src.utils.utils import *

class Trainer(ABC):
    def __init__(self, args, model, optimizer, criterion, features, y_train, y_val, y_test, train_mask, val_mask ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = args.epochs
        self.args = args

        self.torch_features = torch.from_numpy(features.astype(np.float32))
        self.torch_y_train = torch.from_numpy(y_train)
        self.torch_y_val = torch.from_numpy(y_val)
        self.torch_y_test = torch.from_numpy(y_test)
        self.torch_train_mask = torch.from_numpy(train_mask.astype(np.float32))
        self.torch_train_mask_t = torch.transpose(torch.unsqueeze(self.torch_train_mask, 0), 1, 0).repeat(1, y_train.shape[1])
        self.val_mask = val_mask

    def eval(self, features, labels, mask):
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(features).cpu()
            torch_mask = torch.from_numpy(np.array(mask*1., dtype=np.float32))
            torch_mask_t = torch.transpose(torch.unsqueeze(torch_mask, 0), 1, 0).repeat(1, labels.shape[1])
            loss = self.criterion(logits * torch_mask_t, torch.max(labels, 1)[1])
            pred = torch.max(logits, 1)[1]
            accuracy = ((pred == torch.max(labels, 1)[1]).float() * torch_mask_t).sum().item()/ torch_mask.sum().item()
        
        return loss.numpy(), accuracy, pred.numpy(), labels.numpy()
    
    def train(self):
        val_losses = []

        for epoch in range(self.epochs):

            logits = self.model(self.torch_features)
            loss = self.criterion(logits * self.torch_train_mask_t , torch.max(self.torch_y_train, 1)[1])
            accuracy = ((torch.max(logits, 1)[1] == torch.max(self.torch_y_train, 1)[1]).float() * self.torch_train_mask_t).sum().item()/ self.torch_train_mask.sum().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            val_loss, val_acc, pred, labels = self.eval(self.torch_features, self.torch_y_train, self.val_mask)
            val_losses.append(val_loss)

            print("Epoch : {:04d} | Train Loss : {:.4f} | Train Acc : {:.4f} | Val Loss : {:.4f} | Val Acc : {:.4f}".format(epoch+1, loss, accuracy, val_loss, val_acc))

            if epoch > self.args.early_stopping and val_losses[-1] > np.mean(val_losses[-(self.args.early_stopping+1):-1]):
                print("Early stopping...")
                break
        
        print("Optimization Finished")

        



        

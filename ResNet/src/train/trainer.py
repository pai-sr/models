from base.base_trainer import BaseTrainer
from ResNet.src.utils.eval import eval_epoch
import torch
from tqdm import tqdm

class Trainer(BaseTrainer):
    def _train_epoch(self, epoch):
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for images, labels in tqdm(self.train_data_loader):
            x, y = images, labels.long()

            pred = self.model(x)

            self.optimizer.zero_grad()
            loss = self.loss_fn(pred, y)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            acc = self.metric_fn(pred, y)
            epoch_acc += acc

        return (eval_epoch('loss', epoch_loss, self.train_data_loader.dataset.__len__(),
                         self.train_data_loader.batch_size, False)
                , eval_epoch('accuracy', epoch_acc, self.train_data_loader.dataset.__len__(),
                            self.train_data_loader.batch_size))

    def validate(self):
        val_loss = 0
        val_acc = 0

        self.model.eval()

        with torch.no_grad():
            for images, labels in tqdm(self.valid_data_loader):
                x, y = images, labels.long()

                pred = self.model(x)

                loss = self.loss_fn(pred, y)
                val_loss += loss.item()
                acc = self.metric_fn(pred, y)
                val_acc += acc

        return (eval_epoch('loss', val_loss, self.valid_data_loader.dataset.__len__(),
                         self.valid_data_loader.batch_size, False)
                , eval_epoch('accuracy', val_acc, self.valid_data_loader.dataset.__len__(),
                            self.valid_data_loader.batch_size))

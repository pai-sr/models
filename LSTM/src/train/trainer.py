import torch
from tqdm import tqdm
from base.base_trainer import BaseTrainer
from LSTM.src.utils.eval import eval_epoch

class Trainer(BaseTrainer):
    def _train_epoch(self, epoch):
        epoch_loss = 0
        epoch_acc = 0

        self.model.train()

        for x, y in tqdm(self.train_data_loader):
            pred = self.model(x)

            self.optimizer.zero_grad()
            loss = self.loss_fn(pred, y)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        return (eval_epoch('loss', epoch_loss, self.train_data_loader.dataset.__len__(),
                       self.train_data_loader.batch_size, False)
            , eval_epoch('no metric', epoch_acc, self.train_data_loader.dataset.__len__(),
                         self.train_data_loader.batch_size))

    def validate(self):
        val_loss = 0
        val_acc = 0

        self.model.eval()

        with torch.no_grad():
            for x, y in tqdm(self.valid_data_loader):

                pred = self.model(x)

                loss = self.loss_fn(pred, y)
                val_loss += loss.item()

        return (eval_epoch('loss', val_loss, self.valid_data_loader.dataset.__len__(),
                           self.valid_data_loader.batch_size, False)
                , eval_epoch('no metric', val_acc, self.valid_data_loader.dataset.__len__(),
                             self.valid_data_loader.batch_size))
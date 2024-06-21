from base.base_trainer import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, model, loss_fn, optimizer,
                 train_data_loader, logger, *args, **kwargs):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_data_loader = train_data_loader
        self.logger = logger

    def _train_epoch(self, epoch):
        epoch_loss = 0

        self.model.train()

        for idx, batch in enumerate(self.train_data_loader):
            x = batch

            focal_embed, context_embed, focal_bias, context_bias, counts = self.model(x)

            self.optimizer.zero_grad()
            loss = self.loss_fn(focal_embed, context_embed,
                                focal_bias, context_bias, counts,
                                self.model.x_max, self.model.alpha)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            metric = 0

        return (epoch_loss / (self.train_data_loader.dataset.__len__() /
                             self.train_data_loader.batch_size),
               {"no metric" : metric})

    def validate(self):
        """ no validate process in this model"""
        pass
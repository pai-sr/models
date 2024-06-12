from abc import abstractmethod

class BaseTrainer:
    """
    Base class for all trainers
    """

    def __init__(self, model, loss_fn, optimizer, metric_fn,
                 train_data_loader, valid_data_loader, logger, *args, **kwargs):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metric_fn = metric_fn
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.logger = logger

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Train logic for an epoch
        :param epoch: current number of epoch
        :return: trained results
        """
        raise NotImplementedError

    def train(self, epochs):

        for epoch in range(epochs):
            loss, accuracy = self._train_epoch(epoch)
            self.logger.info(f'Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {accuracy}')

        return loss, accuracy

    @abstractmethod
    def validate(self):
        """
        validate logic
        :return: validation results
        """
        raise NotImplementedError


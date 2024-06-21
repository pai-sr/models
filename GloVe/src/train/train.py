import argparse
import logging
import torch.optim as optim

from GloVe.src.dataprovider.data_setter import GloVeDataSetter
from GloVe.src.dataprovider.data_loader import GloVeDataLoader
from GloVe.src.model.model import GloVeModel
from GloVe.src.model.loss import glove_loss
from GloVe.src.train.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(data_path, batch_size, epochs, context_size, embedding_size, min_occurrence, x_max=100, alpha=3/4, lr=0.0001):
    train_ds = GloVeDataSetter(context_size, context_size, min_occurrence, data_path)
    train_dl = GloVeDataLoader(dataset=train_ds, batch_size=batch_size)

    model = GloVeModel(embedding_size, context_size, train_dl.dataset.vocab_size, min_occurrence, x_max, alpha)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = glove_loss

    trainer = Trainer(model=model,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      train_data_loader=train_dl,
                      logger=logger)

    train_loss, train_metrics = trainer.train(epochs=epochs)
    log_msg = f"Train Loss: {train_loss:.4f}"
    log_msg += " | ".join([f'Train {k:} {v}' for k, v in train_metrics.items()])
    logger.info(log_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GloVe Model')
    parser.add_argument("-d", "--data_path", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-c", "--context_size", type=int, default=3)
    parser.add_argument("-emb", "--embedding_size", type=int, default=128)
    parser.add_argument("-m", "--min_occurrence", type=int, default=1)
    parser.add_argument("-x", "--x_max", type=int, default=100)
    parser.add_argument("-a", "--alpha", type=float, default=3/4)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)

    args = parser.parse_args()

    main(args.data_path, args.batch_size,
         args.epochs, args.context_size,
         args.embedding_size, args.min_occurrence,
         args.x_max, args.alpha, args.learning_rate)
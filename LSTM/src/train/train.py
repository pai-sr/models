import argparse
import logging
import torch.optim as optim

from LSTM.src.dataprovider.data_setter import StockDataSetter
from LSTM.src.dataprovider.data_loader import StockDataLoader
from LSTM.src.model.model import LSTMModel, LSTM
from LSTM.src.model.loss import mse_loss
from LSTM.src.train.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(csv_file, epochs=10, batch_size=5000, num_classes=1, input_size=1, hidden_size=2, num_layers=2, seq_length=60, bias=True, lr=0.0001):
    ## DataProvider
    train_ds = StockDataSetter(csv_file=csv_file, seq_length=seq_length)
    train_dl = StockDataLoader(dataset=train_ds, batch_size=batch_size)

    model = LSTM(num_classes=num_classes, input_size=input_size, hidden_size=hidden_size,
                 num_layers=num_layers, seq_length=seq_length, bias=bias)

    def no_metric(output, target):
        return 0

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = mse_loss
    metric_fn = no_metric

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        metric_fn=metric_fn,
        train_data_loader=train_dl,
        valid_data_loader=train_dl,
        logger=logger
    )

    train_loss, train_metrics = trainer.train(epochs=epochs)
    log_msg = f"Train Loss: {train_loss:.4f}"
    log_msg += " | ".join([f'Train {k:} {v}' for k, v in train_metrics.items()])
    logger.info(log_msg)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSTM Model')
    parser.add_argument("-f", "--csv_file", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=5000)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-n", "--num_classes", type=int, default=1)
    parser.add_argument("-i", "--input_size", type=int, default=1)
    parser.add_argument("-hs", "--hidden_size", type=int, default=2)
    parser.add_argument("-l", "--num_layers", type=int, default=2)
    parser.add_argument("-s", "--seq_length", type=int, default=60)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)

    args = parser.parse_args()

    main(args.csv_file, args.epochs, args.batch_size, args.num_classes, args.input_size, args.hidden_size, args.num_layers, args.seq_length, bias=True, lr=args.learning_rate)
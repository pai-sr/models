import argparse
import logging
import torch.optim as optim

from LLama3.src.dataprovider.data_setter import BoolQDataSetter
from LLama3.src.dataprovider.data_loader import BoolQDataLoader
from LLama3.src.model.model import LLamaForSequenceClassification
from LLama3.src.model.loss import cross_entropy_loss
from LLama3.src.model.metric import accuracy
from LLama3.src.train.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(tokenizer_path, seq_len, dim, n_layers, n_heads, n_kv_heads, vocab_size, multiple_of, ffn_dim_multiplier,
                        norm_eps, rope_theta, max_seq_len, batch_size, epochs, lr):
    ## DataProvider
    train_ds = BoolQDataSetter(valid=False, tokenizer_path=tokenizer_path, seq_len=seq_len)
    valid_ds = BoolQDataSetter(valid=True, tokenizer_path=tokenizer_path, seq_len=3)
    train_dl = BoolQDataLoader(dataset=train_ds, batch_size=batch_size)
    valid_dl = BoolQDataLoader(dataset=valid_ds, batch_size=batch_size)

    # Model
    model = LLamaForSequenceClassification(2, dim, n_layers, n_heads, n_kv_heads, vocab_size, multiple_of, ffn_dim_multiplier,
                        norm_eps, rope_theta, max_seq_len)

    ## Optimizer, loss function, metric
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    loss_fn = cross_entropy_loss
    metric_fn = accuracy

    ## Trainer
    trainer = Trainer(model=model,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      metric_fn=metric_fn,
                      train_data_loader=train_dl,
                      valid_data_loader=valid_dl,
                      logger=logger)

    train_loss, train_metrics = trainer.train(epochs)
    log_msg = f"Train Loss: {train_loss:.4f}"
    log_msg += " | ".join([f'Train {k:} {v}' for k, v in train_metrics.items()])
    logger.info(log_msg)

    val_loss, val_metrics = trainer.validate()
    log_msg = f"Validation Loss: {val_loss:.4f}"
    log_msg += " | ".join([f'Validation {k:} {v}' for k, v in val_metrics.items()])
    logger.info(log_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNet Model')
    parser.add_argument("-t", "--tokenizer_path", type=str)
    parser.add_argument("-l", "--seq_len", type=int, default=15)
    parser.add_argument("-d", "--dim", type=int, default=4096)
    parser.add_argument("-nl", "--n_layers", type=int, default=32)
    parser.add_argument("-nh", "--n_heads", type=int, default=32)
    parser.add_argument("-nkvh", "--n_kv_heads", type=int, default=32)
    parser.add_argument("-v", "--vocab_size", type=int, default=-1)
    parser.add_argument("-m", "--multiple_of", type=int, default=256)
    parser.add_argument("-dm", "--ffn_dim_multiplier", type=float)
    parser.add_argument("-ne", "--norm_eps", type=float, default=1e-5)
    parser.add_argument("-rt", "--rope_theta", type=float, default=50000.0)
    parser.add_argument("-ml", "--max_seq_len", type=int, default=2048)
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)

    args = parser.parse_args()

    main(args.tokenizer_path, args.seq_len, args.dim, args.n_layers, args.n_heads, args.n_kv_heads,
         args.vocab_size, args.multiple_of, args.ffn_dim_multiplier, args.norm_eps, args.rope_theta, args.max_seq_len,
         args.batch_size, args.epochs, args.learning_rate)


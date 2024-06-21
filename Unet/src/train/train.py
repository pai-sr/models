import argparse
import logging
import torch.optim as optim

from Unet.dataprovider.data_loader import COCO
from Unet.model import model, loss, metric
from Unet.train.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(device, data_dir, batch_size, partition, epochs, num_classes, lr, freeze_bn):
    ## DataProvider
    train_dl = COCO(data_dir=data_dir, batch_size=batch_size,
                    split='train', partition=partition)
    valid_dl = COCO(data_dir=data_dir, batch_size=batch_size,
                  split='val', partition=partition)

    ## model
    model_ = model.UNet(num_classes=num_classes, freeze_bn=freeze_bn)

    ## Optimizer, loss function, metric
    optimizer = optim.SGD(model_.parameters(), lr=lr)
    loss_fn = loss.CrossEntropyLoss2d()
    metric_fn = metric.eval_metrics

    trainer = Trainer(model=model_,
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
    parser.add_argument("-m", "--model_arch", type=str, default="resnet34", choices=["resnet34", "resnet50"])
    parser.add_argument("-b", "--batch_size", type=int, default=64)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-n", "--num_classes", type=int, default=10)
    parser.add_argument("-ic", "--in_channels", type=int, default=1)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-4)

    args = parser.parse_args()

    main(args.model_arch, args.batch_size, args.epochs, args.num_classes, args.in_channels, args.learning_rate)

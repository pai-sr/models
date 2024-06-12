import argparse
import logging
import torch.nn as nn
import torch.optim as optim

from torchvision.transforms import ToTensor

from ResNet.src.dataprovider.data_setter import MnistDataSetter
from ResNet.src.dataprovider.data_loader import MnistDataLoader
from ResNet.src.model.ResNet.model import ResNet, BasicBlock, Bottleneck
from ResNet.src.model.ResNet.metric import accuracy
from ResNet.src.train.trainer import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(model_arch, batch_size=64, epochs=10, num_classes=10, in_channels=1, lr=3e-4):
    ## DataProvider
    train_ds = MnistDataSetter("mnist", train=True, download=True, transform=ToTensor())
    valid_ds = MnistDataSetter("mnist", train=False, download=True, transform=ToTensor())
    train_dl = MnistDataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = MnistDataLoader(valid_ds, batch_size=batch_size, shuffle=True)

    ## Model
    if model_arch == "resnet34":
        model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)
    if model_arch == "resnet50":
        model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, in_channels=in_channels)

    ## Optimizer, loss function, metric
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = accuracy

    ## Trainer
    trainer = Trainer(model=model,
                      loss_fn=loss_fn,
                      optimizer=optimizer,
                      metric_fn=metric_fn,
                      train_data_loader=train_dl,
                      valid_data_loader=valid_dl,
                      logger=logger)

    train_loss, train_acc = trainer.train(epochs)
    logger.info(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc}")

    val_loss, val_acc = trainer.validate()
    logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc}")

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
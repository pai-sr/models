import time
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from base.base_trainer import BaseTrainer
from Unet.src.model.metric import AverageMeter, eval_metrics

class Trainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(Trainer, self).__init__(*args, **kwargs)
        self.__dict__.update(self.kwargs)

    def _train_epoch(self, epoch):
        self.model.to(self.device)
        self.model.train()

        if self.freeze_bn:
            if isinstance(self.model, nn.DataParallel):
                self.model.module.freeze_bn()
            else:
                self.model.freeze_bn()

        tic = time.time()
        self._reset_metrics()
        tbar = tqdm(self.train_data_loader, ncols=130)
        for batch_idx, (data, target) in enumerate(tbar):
            self.data_time.update(time.time() - tic)
            data, target = data.to(self.device), target.to(self.device)

            pred = self.model(data)

            self.optimizer.zero_grad()
            loss = self.loss_fn(pred, target)
            if isinstance(self.loss_fn, nn.DataParallel):
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            self.total_loss.update(loss.item())

            seg_metrics = eval_metrics(pred, target, self.model.num_classes)
            self._update_seg_metrics(*seg_metrics)
            pixAcc, mIoU, _ = self._get_seg_metrics().values()

        return self.total_loss.average, self._get_seg_metrics()

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.total_inter, self.total_union = 0, 0
        self.total_correct, self.total_label = 0, 0

    def _update_seg_metrics(self, correct, labeled, inter, union):
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def _get_seg_metrics(self):
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return {
            "pixel_accuracy": np.round(pixAcc, 3),
            "mean_iou": np.round(mIoU, 3),
            "class_iou": dict(zip(range(self.model.num_classes), np.round(IoU, 3)))
        }
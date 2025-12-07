import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torchmetrics

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(pl.LightningModule):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512], lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
        self.train_metric = torchmetrics.classification.BinaryStatScores()
        self.val_metric = torchmetrics.classification.BinaryStatScores()
        self.test_metric = torchmetrics.classification.BinaryStatScores()
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], 
                                               mode='bilinear', align_corners=True)

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)
    
    def _compute_dice(self, preds, targets):
        smooth = 1e-6
        preds = (preds > 0.5).float()
        
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice
    
    def _shared_step(self, batch, batch_idx, stage):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        
        y_prob = torch.sigmoid(y_hat)
        
        metric = getattr(self, f"{stage}_metric")
        metric.update(y_prob, y.long())
        
        y_pred = (y_prob > 0.5).float()
        dice = self._compute_dice(y_pred, y)
        
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_dice", dice, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch
        y_hat = self(x)
        return torch.sigmoid(y_hat)
    
    def on_train_epoch_end(self):
        scores = self.train_metric.compute()
        tp, fp, tn, fn, sup = scores

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        
        self.log("train_dice_epoch", dice, prog_bar=True)
        self.train_metric.reset()
        
    def on_validation_epoch_end(self):
        scores = self.val_metric.compute()
        tp, fp, tn, fn, sup = scores

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        
        self.log("val_dice_epoch", dice, prog_bar=True)
        self.val_metric.reset()
        
    def on_test_epoch_end(self):
        scores = self.test_metric.compute()
        tp, fp, tn, fn, sup = scores
        
        dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
        
        self.log("test_dice_epoch", dice, prog_bar=True)
        self.test_metric.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }
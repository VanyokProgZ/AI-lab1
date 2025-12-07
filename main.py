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

from unet import UNet
from data import prepare_data, SegmentationDataset

LR =1e-5
ERS = 1e-4

def main():
    mlflow_logger = MLFlowLogger(
        experiment_name="unet_segmentation",
        tracking_uri="mlruns",
        log_model="all"
    )
    
    data_splits = prepare_data("df.csv")
    
    train_dataset = SegmentationDataset(
        data_splits['train'][0], 
        data_splits['train'][1], 
        transform_mode="train"
    )
    val_dataset = SegmentationDataset(
        data_splits['val'][0], 
        data_splits['val'][1], 
        transform_mode=None
    )
    test_dataset = SegmentationDataset(
        data_splits['test'][0], 
        data_splits['test'][1], 
        transform_mode=None
    )

    batch_size = 4
    num_workers = min(4, torch.multiprocessing.cpu_count())
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    model = UNet(in_channels=3, out_channels=1, lr=LR)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="unet-{epoch:03d}-{val_dice_epoch:.3f}",
        monitor="val_dice_epoch",
        mode="max",
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    
    early_stopping_callback = EarlyStopping(
        monitor="val_dice_epoch",
        patience=15,
        mode="max",
        verbose=True,
        min_delta=ERS
    )
    
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",
        logger=mlflow_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=10,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,
        accumulate_grad_batches=1,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm"
    )
    
    mlflow_logger.log_hyperparams({
        "batch_size": batch_size,
        "learning_rate": LR,
        "architecture": "UNet",
        "dataset": "Segmentation",
        "epochs": 100,
        "features": [64, 128, 256, 512]
    })
    
    trainer.fit(
        model, 
        train_dataloaders=train_loader, 
        val_dataloaders=val_loader
    )
    
    if checkpoint_callback.best_model_path:
        test_results = trainer.test(
            model, 
            dataloaders=test_loader, 
            ckpt_path="best"
        )
    else:
        test_results = trainer.test(model, dataloaders=test_loader)

    mlflow.pytorch.log_model(
        model,
        "unet_model",
        registered_model_name="UNet lab"
    )
    
    print(f"\n{'='*50}")
    print(f"Закончил обучаться!")
    if test_results:
        print(f"Результаты тестов:")
        for key, value in test_results[0].items():
            print(f"  {key}: {value:.4f}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
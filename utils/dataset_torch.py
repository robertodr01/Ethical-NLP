from tkinter import TRUE
from typing import Tuple
import pandas as pd
import lightning as l
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from transformers import AutoTokenizer, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup, AutoModel
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from lightning.pytorch.cli import LightningCLI

LABEL_COLUMNS = ['Morally Negative',
                'Morally Positive',
                'Neutral',
                'Neutral but Negative Sentiment',
                'Neutral but Positive Sentiment',
                'Partially Negative',
                'Partially Neutral',
                'Partially Positive']

def get_splits(n_instances: int, train_split_percentage: float, val_split_percentage: float) -> Tuple[int, int, int]:
    """
    Calculate dataset splits based on specified percentages.

    Args:
        n_instances (int): Total number of instances.
        train_split_percentage (float): Percentage of instances for the training split.
        val_split_percentage (float): Percentage of instances for the validation split.

    Returns:
        Tuple[int, int, int]: Number of instances for training, validation, and test splits.
    """
    train_split = int(n_instances * train_split_percentage / 100)
    remaining_split = n_instances - train_split
    test_split = remaining_split - int(n_instances * val_split_percentage / 100)
    val_split = remaining_split - test_split

    # If no test set is required, then test_split is just remainder, that we can add to the train
    if train_split_percentage + val_split_percentage >= 100.0:
        train_split = train_split + test_split
        test_split = 0

    return train_split, val_split, test_split


class CustomDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer,  max_token_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len =  max_token_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        row = self.data.iloc[index]

        text = row.text
        labels = row[LABEL_COLUMNS]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = self.max_token_len,
            return_token_type_ids = False,
            padding = "max_length", 
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        return dict(
            text=text,
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.FloatTensor(labels)
        )



class CustomDataModule(l.LightningDataModule):

    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: AutoTokenizer,
                 max_token_len = 128,
                 train_split_percentage: float = 70,
                 val_split_percentage: float = 10,
                 batch_size: int = 512,
                 num_workers: int = 0,
                 shuffle: bool = False) -> None:
        
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.train_split_percentage = train_split_percentage
        self.val_split_percentage = val_split_percentage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.max_token_len = max_token_len
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: str) -> None:

        train_sets = []
        val_sets = []
        test_sets = []

        dataset = CustomDataset(self.data, self.tokenizer, self.max_token_len)

        train_split, val_split, test_split = get_splits(n_instances=len(dataset),
                                                                    train_split_percentage=self.train_split_percentage,
                                                                    val_split_percentage=self.val_split_percentage)
        
        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_split, val_split, test_split])

        # print(self.train_set, len(self.train_set))
        # print(self.val_set, len(self.val_set))
        # print(self.test_set, len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers,
                          drop_last=True, shuffle=self.shuffle)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers,
                          drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers,
                          drop_last=True)


class Model(l.LightningModule):

    def __init__(self, bert_model_name: str, lr: float, weigth_decay: float, n_warmup_steps = None):

        super().__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name, return_dict=True)
        self.classifier = nn.Linear(self.bert.config.hidden_size, len(LABEL_COLUMNS))
        self.n_warmup_steps = n_warmup_steps
        self.lr = lr
        self.weigth_decay = weigth_decay
        self.criterion = nn.CrossEntropyLoss()
        self.weighted_accuracy = MulticlassAccuracy(num_classes=len(LABEL_COLUMNS), average='weighted')
        self.weighted_precision = MulticlassPrecision(num_classes=len(LABEL_COLUMNS), average='weighted')
        self.weighted_recall = MulticlassRecall(num_classes=len(LABEL_COLUMNS), average='weighted')
        self.weighted_f1 = MulticlassF1Score(num_classes=len(LABEL_COLUMNS), average='weighted')
        self.save_hyperparameters()

    def configure_optimizers(self):

        optimizer = torch.optim.AdamW(params = self.bert.parameters(),
                                      lr = self.lr,
                                      weight_decay= self.weigth_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer= optimizer)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler
            }
        }

    def forward(self, input_ids, attention_mask, labels=None):

        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        output = torch.sigmoid(output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
            
        return loss, output


    def training_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)

        accuracy = self.weighted_accuracy(outputs, labels)
        precision = self.weighted_precision(outputs, labels)
        recall = self.weighted_recall(outputs, labels)
        f1 = self.weighted_f1(outputs, labels)

        # self.log("train_loss", loss, prog_bar=True, logger=True)

        self.log_dict({"train_loss": loss, "train_acc": accuracy, 
                       "train_prec": precision, "train_rec": recall,
                       "train_f1": f1}, on_step=True, on_epoch=True, prog_bar=True, logger = True)

        return {"loss": loss, "predictions": outputs, "labels": labels}

    
    def validation_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        # self.log("val_loss", loss, prog_bar=True, logger=True)

        accuracy = self.weighted_accuracy(outputs, labels)
        precision = self.weighted_precision(outputs, labels)
        recall = self.weighted_recall(outputs, labels)
        f1 = self.weighted_f1(outputs, labels)

        # self.log("train_loss", loss, prog_bar=True, logger=True)

        self.log_dict({"val_loss": loss, "val_acc": accuracy, 
                       "val_prec": precision, "val_rec": recall,
                       "val_f1": f1}, on_step=False, on_epoch=True, prog_bar=True, logger = True)
        
        return {"loss": loss, "predictions": outputs, "labels": labels}

    def test_step(self, batch, batch_idx):

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        # self.log("test_loss", loss, prog_bar=True, logger=True)

        accuracy = self.weighted_accuracy(outputs, labels)
        precision = self.weighted_precision(outputs, labels)
        recall = self.weighted_recall(outputs, labels)
        f1 = self.weighted_f1(outputs, labels)

        # self.log("train_loss", loss, prog_bar=True, logger=True)

        self.log_dict({"test_loss": loss, "test_acc": accuracy, 
                       "test_prec": precision, "test_rec": recall,
                       "test_f1": f1}, on_step=False, on_epoch=True, prog_bar=True, logger = True)
        
        return {"loss": loss, "predictions": outputs, "labels": labels}
    
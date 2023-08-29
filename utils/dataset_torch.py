from mimetypes import init
import pandas as pd
import lightning as l
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, default_collate
from transformers import AutoTokenizer, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1_score, precision, recall

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

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        row = self.data.iloc[index]

        text = row.text
        label = row.target

        return text, label

        # encoding = self.tokenizer.encode_plus(
        #     text,
        #     add_special_tokens = True,
        #     max_length = self.max_token_length,
        #     return_token_type_ids = False,
        #     padding = "max_length", 
        #     truncation = True,
        #     return_attention_mask = True,
        #     return_tensors = 'pt'
        # )

        # return dict(
        #     text,
        #     input_ids = encoding['input_ids'].flatten(),
        #     attention_mask = encoding['attention_mask'].flatten(),
        #     label = torch.FloatTensor(label)
        # )



class CustomDataModule(l.LightningDataModule):

    def __init__(self,
                 data: pd.DataFrame,
                 tokenizer: BertTokenizer,
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
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage: str) -> None:

        train_sets = []
        val_sets = []
        test_sets = []

        dataset = CustomDataset(self.data)

        train_split, val_split, test_split = get_splits(n_instances=len(dataset),
                                                                    train_split_percentage=self.train_split_percentage,
                                                                    val_split_percentage=self.val_split_percentage)
        
        self.train_set, self.val_set, self.test_set = random_split(dataset, [train_split, val_split, test_split])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers,
                          drop_last=True, shuffle=self.shuffle, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers,
                          drop_last=True, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers,
                          drop_last=True, collate_fn=self.collate_fn)

    def collate_fn(self, examples):

        text, target = default_collate(examples)

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = self.max_token_length,
            return_token_type_ids = False,
            padding = "max_length", 
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )

        return dict(
            text,
            input_ids = encoding['input_ids'].flatten(),
            attention_mask = encoding['attention_mask'].flatten(),
            labels = torch.FloatTensor(labels)
        )


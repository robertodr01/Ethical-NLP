from mimetypes import init
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1_score, precision, recall


class CustomDataset(Dataset):

    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_length: int = 128):
        self.tokenizer = tokenizer
        self.data = data
        self.max_token_length = max_token_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        row = self.data.iloc[index]

        text = row.text
        label = row.target

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
            label = torch.FloatTensor(label)
        )
      


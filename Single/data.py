import os
import json
import torch
import litgpt
from litgpt.data import SFTDataset
from torch.utils.data import DataLoader, random_split
import numpy as np

import json
from dataclasses import dataclass, field
from typing import Optional, Union
from pathlib import Path
from lightning import LightningDataModule
from litgpt import Tokenizer
from litgpt.data import Alpaca2k, SFTDataset
from litgpt.prompts import PromptStyle
from datasets import load_dataset, concatenate_datasets
from torch.utils.data.distributed import DistributedSampler

class LLMData:
    def __init__(self, repo_id, val_split_fraction=0.05, batch_size=1, max_seq_length=512, num_workers=4):
        self.repo_id = repo_id
        self.val_split_fraction = val_split_fraction
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.num_workers = num_workers
        self.tokenizer = litgpt.Tokenizer(f"checkpoints/TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    def prepare_data(self):
        datasets = []
        for repo in self.repo_id:
            datasets.append(load_dataset(repo)["train"])
        dataset = concatenate_datasets(datasets)
        train_val_split = dataset.train_test_split(test_size=self.val_split_fraction)
        train_data = train_val_split["train"]
        val_data = train_val_split["test"]

        self.train_dataset = SFTDataset(
            data=train_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            prompt_style="alpaca"
        )
        self.val_dataset = SFTDataset(
            data=val_data,
            tokenizer=self.tokenizer,
            max_seq_length=self.max_seq_length,
            prompt_style="alpaca"
        )

    def get_dataloaders(self, world_size, rank):
        train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, num_workers=self.num_workers)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, sampler=val_sampler, num_workers=self.num_workers)
        return train_loader, val_loader
    
data = LLMData(repo_id=["mhenrichsen/alpaca_2k_test", "databricks/databricks-dolly-15k"], val_split_fraction=0.2)
data.prepare_data()

train_loader, val_loader = data.get_dataloaders(world_size=1, rank=0)

for line in train_loader:
    print(line)
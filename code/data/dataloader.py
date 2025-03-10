# data/dataloader.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
from config import NUM_CPU

class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def get_dataloaders(train_split, val_split, tokenizer, batch_size=32, num_workers=NUM_CPU):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_dataset = DatasetWrapper(train_split)
    val_dataset = DatasetWrapper(val_split)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=data_collator
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=data_collator
    )
    return train_loader, val_loader

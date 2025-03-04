from abc import ABC, abstractmethod
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from config import NUM_CPU

class BaseDataset(ABC):
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data = None
        self.load_data()

    def load_data(self):
        if self.filepath:
            self.data = load_from_disk(self.filepath)
        else:
            self.data = self.load_dataset()

    def clean_and_split(self, datasets_list, min_len, test_size):
        filtered_datasets = []
        for ds in datasets_list:
            ds_filtered = ds.filter(lambda example: len(example["tokens"]) > min_len, num_proc=NUM_CPU)
            for split in ds_filtered.keys():
                filtered_datasets.append(ds_filtered[split])
        
        concat_dataset = concatenate_datasets(filtered_datasets)
        final_dataset = concat_dataset.shuffle(seed=42)
        split_train_rest = final_dataset.train_test_split(test_size=test_size, seed=42)
        split_val_test = split_train_rest["test"].train_test_split(test_size=0.5, seed=42)
        
        return DatasetDict({
            "train": split_train_rest["train"],
            "validation": split_val_test["train"],
            "test": split_val_test["test"],
        })
    
    @abstractmethod
    def load_dataset(self):
        pass
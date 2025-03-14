from abc import ABC, abstractmethod
from datasets import load_from_disk, concatenate_datasets, DatasetDict
from config import NUM_CPU
from .data_utils import split_dataset

class BaseDataset(ABC):
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data = None
        self.load_data()
        
    def load_data(self):
        if self.filepath:
            self.data = load_from_disk(self.filepath)
        else:
            self.data = self._load_dataset()

    def clean_and_split(self, datasets_list, min_len, test_size, seed=42):
        filtered_datasets = []
        for ds in datasets_list:
            ds_filtered = ds.filter(lambda example: len(example["tokens"]) > min_len, num_proc=NUM_CPU)
            for split in ds_filtered.keys():
                filtered_datasets.append(ds_filtered[split])
        
        concat_dataset = concatenate_datasets(filtered_datasets).shuffle(seed=seed)
        return split_dataset(concat_dataset, test_size, seed)
    
    @abstractmethod
    def _load_dataset(self):
        pass

    @abstractmethod
    def get_entities(self):
        pass

    @abstractmethod
    def get_label_mapping(self):
        pass
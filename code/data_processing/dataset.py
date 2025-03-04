from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk, DatasetDict, concatenate_datasets
import os

NUM_CPU = os.cpu_count()
DEFAULT_LABELS = {
    "O"   : "Outside",
    "DISO": "Trouble",
    "CHEM": "Produit Chimique",
    "PROC": "Procédure",
    "LIVB": "Être vivant",
    "ANAT": "Anatomie",
    "PHYS": "Physiologie",
    "OBJC": "Objet",
    "DEVI": "Dispositif",
    "GEOG": "Géographie",
    "PHEN": "Phénomène"
}

class BaseDataset(ABC):
    def __init__(self, filepath=None, tokenizer_name=None):
        self.filepath = filepath
        self.tokenizer = self._load_tokenizer(tokenizer_name) if tokenizer_name else None
        self.data = None

    def _load_tokenizer(self, tokenizer_name: str):
        try:
            return AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception:
            raise ValueError(f"The tokenizer '{tokenizer_name}' could not be loaded. Please check if it exists.")

    def set_tokenizer(self, tokenizer_name: str):
        self.tokenizer = self._load_tokenizer(tokenizer_name)

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def tokenize_and_align_labels(self, examples: dict):
        pass

class QUAERODataset(BaseDataset):
    def __init__(self, split_names=None, min_len=5, test_size=0.2, filepath=None, tokenizer_name=None):
        super().__init__(filepath, tokenizer_name)
        self.split_names = split_names if split_names is not None else ["emea"]
        self.min_len = min_len
        self.test_size = test_size
        self.load_data()

    def load_data(self):
        if self.filepath:
            self.data = load_from_disk(self.filepath)
        else:
            self.data = self.load_quaero_dataset()

    def load_quaero_dataset(self):
        datasets_list = []
        for split_name in self.split_names:
            ds = load_dataset("DrBenchmark/QUAERO", split_name, trust_remote_code=True)
            ds_filtered = ds.filter(lambda example: len(example["tokens"]) > self.min_len, num_proc=NUM_CPU)
            for split in ds_filtered.keys():
                datasets_list.append(ds_filtered[split])
    
        concat_dataset = concatenate_datasets(datasets_list)
        final_dataset = concat_dataset.shuffle(seed=42)
        split_train_rest = final_dataset.train_test_split(test_size=self.test_size, seed=42)
        split_val_test = split_train_rest["test"].train_test_split(test_size=0.5, seed=42)
    
        return DatasetDict({
            "train": split_train_rest["train"],
            "validation": split_val_test["train"],
            "test": split_val_test["test"],
        })

    def tokenize_and_align_labels(self, examples: dict):
        if not self.tokenizer:
            raise ValueError("Tokenizer is not defined. Please define it before tokenizing.")
            
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=True, padding=True, is_split_into_words=True)
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx] if self.tokenizer.is_fast else -100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_dataset(self):
        return {
            split: dataset.map(self.tokenize_and_align_labels, batched=True, num_proc=NUM_CPU)
            for split, dataset in self.data.items()
        }
    
    def get_entities(self, group: bool = True):
        entities = set()
        for split in self.data:
            curr_entities = self.data[split].features["ner_tags"].feature.names
            if group:
                curr_entities = {entity[2:] if entity.startswith(("B-", "I-")) else entity for entity in curr_entities}
            entities.update(curr_entities)
        return list(entities)

    def get_label_mapping(self):
        existing_entities = set(self.get_entities())
        return {
            entity: DEFAULT_LABELS.get(entity, "Unknown")
            for entity in existing_entities
        }

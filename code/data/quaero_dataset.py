from datasets import load_dataset
from .base_dataset import BaseDataset
from .data_utils import DEFAULT_LABELS

class DrBenchmarkQUAERO(BaseDataset):
    def __init__(self, split_names=None, min_len=5, test_size=0.1, filepath=None):
        self.split_names = split_names if split_names is not None else ["emea"]
        self.min_len = min_len
        self.test_size = test_size
        super().__init__(filepath)

    def _load_dataset(self):
        datasets_list = [load_dataset("DrBenchmark/QUAERO", split_name, trust_remote_code=True) for split_name in self.split_names]
        return self.clean_and_split(datasets_list, self.min_len, self.test_size)

    def get_entities(self, group: bool=False):
        entities = self.data["train"].features["ner_tags"].feature.names
        if group:
            entities = [
                entity[2:] if entity.startswith(("B-", "I-")) else entity
                for entity in entities
            ]
        return entities

    def get_label_mapping(self):
        existing_entities = set(self.get_entities(group=True))
        return {
            entity: DEFAULT_LABELS.get(entity, "Unknown")
            for entity in existing_entities
        }
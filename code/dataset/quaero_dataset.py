from datasets import load_dataset
from .base_dataset import BaseDataset

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

class QUAERODataset(BaseDataset):
    def __init__(self, split_names=None, min_len=5, test_size=0.2, filepath=None):
        self.split_names = split_names if split_names is not None else ["emea"]
        self.min_len = min_len
        self.test_size = test_size
        super().__init__(filepath)

    def load_dataset(self):
        datasets_list = [load_dataset("DrBenchmark/QUAERO", split_name, trust_remote_code=True) for split_name in self.split_names]
        return self.clean_and_split(datasets_list, self.min_len, self.test_size)

    def get_entities(self, group: bool = True):
        entities = set()
        for split in self.data:
            curr_entities = self.data[split].features["ner_tags"].feature.names
            if group:
                curr_entities = {
                    entity[2:] if entity.startswith(("B-", "I-")) else entity
                    for entity in curr_entities
                }
            entities.update(curr_entities)
        return list(entities)

    def get_label_mapping(self):
        existing_entities = set(self.get_entities())
        return {
            entity: DEFAULT_LABELS.get(entity, "Unknown")
            for entity in existing_entities
        }
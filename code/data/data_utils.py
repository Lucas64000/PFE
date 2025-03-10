from datasets import DatasetDict, concatenate_datasets, load_dataset
from .base_dataset import get_entities
import json 
import os

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


def flatten_dataset(dataset):
    if isinstance(dataset, DatasetDict):
        return concatenate_datasets([dataset[split] for split in dataset.keys()])
    return dataset


def split_dataset(dataset, test_ratio, seed=42):
    full_split = dataset.train_test_split(test_size=2 * test_ratio, seed=seed)
    val_test_split = full_split["test"].train_test_split(test_size=0.5, seed=seed)
    return DatasetDict({
        "train": full_split["train"],
        "validation": val_test_split["train"],
        "test": val_test_split["test"]
    })


def remove_tokens_with_O(dataset_dict: DatasetDict, threshold: float = 0.9) -> DatasetDict:
    """
    Supprime les entrées où le pourcentage de 'O' (0) dans ner_tags dépasse un certain seuil.

    :param dataset_dict: DatasetDict contenant les datasets train, validation, test.
    :param threshold: Pourcentage maximal de 'O' pour retirer la ligne (ex: 0.9 pour 90%).
    :return: DatasetDict filtré.
    """
    
    def filter_function(example):
        ner_tags = example["ner_tags"]
        return (sum(tag == 0 for tag in ner_tags) / len(ner_tags)) < threshold

    # Appliquer le filtrage sur chaque split
    filtered_datasets = {split: dataset.filter(filter_function) for split, dataset in dataset_dict.items()}
    
    return DatasetDict(filtered_datasets)


def merge_datasets(dataset_original, *additional_datasets, test_ratio=0.1, seed=42):
    """
    Fusionne un ou plusieurs DatasetDict/Dataset en un seul, mélange les données et les re-sépare en train/val/test.

    :param dataset_original: Un DatasetDict ou un Dataset contenant les données de base.
    :param additional_datasets: Autres DatasetDict/Dataset à fusionner avec dataset_original.
    :param test_ratio: Proportion des données pour le test (par défaut 10%).
    :param seed: Graine aléatoire pour le mélange et le split (par défaut 42).
    :return: Le Dataset original avec un split choisi (mais ratio test = ratio validation), et un DatasetDict plus gros avec "train", "validation" et le même "test" que le DatasetDict original.
    """

    assert 0 < test_ratio < 0.5, "test_ratio doit être dans l'intervalle (0, 0.5) pour un split valide."

    # Fusionner dataset_original s'il s'agit d'un DatasetDict
    dataset_fusionned = flatten_dataset(dataset_original)
    dataset_fusionned = dataset_fusionned.shuffle(seed=seed)

    # Séparer en train (val + test)
    split_train_valtest_base = dataset_fusionned.train_test_split(test_size=2 * test_ratio, seed=seed)
    split_val_test_base = split_train_valtest_base["test"].train_test_split(test_size=0.5, seed=seed)

    # Fusionner les datasets additionnels avec le train original
    datasets_to_merge = [split_train_valtest_base["train"], split_val_test_base["train"]]

    for dataset in additional_datasets:
        dataset = flatten_dataset(dataset)
        datasets_to_merge.append(dataset)

    dataset_rest = concatenate_datasets(datasets_to_merge).shuffle(seed=seed)

    # Calcul du ratio de validation pour garder la même proportion que dans dataset_base
    val_ratio = len(split_val_test_base["test"]) / len(dataset_rest)

    # Séparer en train et validation
    split_train_val = dataset_rest.train_test_split(test_size=val_ratio, seed=seed)

    # Créer le DatasetDict final à partir de l'original
    dataset_base = DatasetDict({
        "train": split_train_valtest_base["train"],
        "validation": split_val_test_base["train"],
        "test": split_val_test_base["test"],
    })

    # Créer le DatasetDict final de la fusion
    dataset_full = DatasetDict({
        "train": split_train_val["train"],
        "validation": split_train_val["test"],
        "test": split_val_test_base["test"],
    })

    return dataset_base, dataset_full


def group_ner_tags(ner_tags, label_list):
    groups = []
    current_group = None
    
    for i, tag in enumerate(ner_tags):
        if isinstance(tag, int):
            tag = label_list[tag]
        
        if tag == "O":
            if current_group is not None:
                groups.append(current_group)
                current_group = None
            continue
        
        if tag.startswith("B-") or tag.startswith("I-"):
            tag_clean = tag[2:]
        else:
            tag_clean = tag
        
        tag_full = DEFAULT_LABELS.get(tag_clean, tag_clean)
        
        if current_group is None:
            current_group = [i, i, tag_full]
        else:
            
            if current_group[2] == tag_full:
                current_group[1] = i
            else:
                groups.append(current_group)
                current_group = [i, i, tag_full]
    
    if current_group is not None:
        groups.append(current_group)
    
    return groups


def convert_data_to_json(dataset, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    annotated_data = {}
    label_list = dataset.get_entities(group=False)
    data = dataset.data
    if isinstance(data, DatasetDict):  
        for split, subset in data.items():
            annotated_data[split] = process_dataset(subset, label_list)
            file_path = os.path.join(output_folder, f"{split}.json")
            save_json(annotated_data[split], file_path)
    else:
        annotated_data = process_dataset(data, label_list)
        file_path = os.path.join(output_folder, "data.json")
        save_json(annotated_data, file_path)


def process_dataset(subset, label_list):
    return [
        {
            "tokenized_text": example["tokens"],
            "ner_tags": group_ner_tags(example["tokens"], example["ner_tags"], label_list),
        }
        for example in subset
    ]


def save_json(data, file_path):
    with open(file_path, "wt") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def load_json_dataset(input_path):
    if os.path.isdir(input_path):
        data = {}
        for file in os.listdir(input_path):
            if file.endswith(".json"):
                key = os.path.splitext(file)[0]
                path = os.path.join(input_path, file)
                with open(path, "r") as f:
                    data[key] = json.load(f)
        if not data:
            raise ValueError("No JSON files found in the folder.")
        
    else:
        with open(input_path, "r") as f:
            data = json.load(f)
    
    return data

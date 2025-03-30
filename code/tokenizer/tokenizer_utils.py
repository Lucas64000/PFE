from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset
from config import NUM_PROC


def load_tokenizer(tokenizer_name: str, **kwargs):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
    except Exception as e:
        raise ValueError(f"The tokenizer '{tokenizer_name}' could not be loaded. Please check if it exists.")
    return tokenizer


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=258,
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


def tokenize_dataset(dataset: DatasetDict, tokenizer, batch_size: int = 1000):
    def tokenize_function(examples):
        return tokenize_and_align_labels(examples, tokenizer)

    if isinstance(dataset, DatasetDict):
        first_split = next(iter(dataset.keys()))
        remove_columns = dataset[first_split].column_names
    elif isinstance(dataset, Dataset):
        remove_columns = dataset.column_names
    else:
        raise ValueError("L'entrée doit être un Dataset ou un DatasetDict.")

    return dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=NUM_PROC,
        remove_columns=remove_columns,
    )
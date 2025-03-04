from transformers import AutoTokenizer
from datasets import DatasetDict
from config import NUM_CPU

def load_tokenizer(tokenizer_name: str, **kwargs):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
    except Exception as e:
        raise ValueError(f"The tokenizer '{tokenizer_name}' could not be loaded. Please check if it exists.")
    return tokenizer

def tokenize_and_align_labels(examples: dict, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        is_split_into_words=True
    )
    
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
                label_ids.append(label[word_idx] if tokenizer.is_fast else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def tokenize_dataset(dataset: DatasetDict, tokenizer, batch_size: int = 1000):
    def tokenize_function(examples):
        return tokenize_and_align_labels(examples, tokenizer)
    
    return dataset.map(
        tokenize_function,
        batched=True,
        batch_size=batch_size,
        num_proc=NUM_CPU,
        remove_columns=["tokens", "ner_tags"]
    )

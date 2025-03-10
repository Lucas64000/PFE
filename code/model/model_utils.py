from transformers import AutoModelForTokenClassification

def load_hf_model(checkpoint: str, label_list: str):
    num_labels = len(label_list)
    id2label = {i: tag for i, tag in enumerate(label_list)}
    label2id = {tag: i for i, tag in enumerate(label_list)}
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )

    return model

def load_from_dir(checkpoint: str):
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        trust_remote_code=True,
    )

    return model
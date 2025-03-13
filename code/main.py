from tokenizer.tokenizer_utils import load_tokenizer, tokenize_dataset
from dataset.quaero_dataset import QUAERODataset

if __name__ == "__main__":
    dataset = QUAERODataset()
    path = "PantagrueLLM/jargon-biomed"
    tokenizer = load_tokenizer(path)
    tokenized_ds = tokenize_dataset(dataset.data, tokenizer)
    print(tokenized_ds)
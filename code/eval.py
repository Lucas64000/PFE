import argparse
import os
import yaml
from datasets import load_from_disk
from data.quaero_dataset import DrBenchmarkQUAERO
from model.model_utils import load_hf_model
from tokenizer.tokenizer_utils import load_tokenizer
from training.evaluator import ModelEvaluator
from model.model_utils import load_from_dir


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML config file")
    parser.add_argument("--normalize", action="store_true", help="Normalize confusion matrix")
    return parser

def load_config_as_namespace(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    config = load_config_as_namespace(args.config)

    dataset_path = config.dataset["dataset_path"]
    save_path = config.evaluator["save_path"]

    if os.path.exists(dataset_path):
        print(f"Chargement du dataset d'évaluation depuis : {dataset_path}")
        dataset = load_from_disk(dataset_path=dataset_path)
        eval_dataset = dataset.get("validation")
    else:
        print("Le chemin du dataset d'évaluation est invalide ou inexistant.")
        exit(1)

    checkpoint = config.model["checkpoint"]
    base_path = checkpoint.split("/checkpoint-")[0]

    model = load_from_dir(checkpoint)
    tokenizer = load_tokenizer(base_path)

    evaluator = ModelEvaluator(
        model=model,
        dataset=eval_dataset,
        tokenizer=tokenizer,
        output_dir=save_path,
    )

    evaluator.evaluate(normalize=args.normalize)

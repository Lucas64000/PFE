import argparse
import yaml
from data.quaero_dataset import DrBenchmarkQUAERO
from model.model_utils import load_hf_model
from training.trainer import ModelTrainer
from transformers import TrainingArguments

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml", help="Config path")
    parser.add_argument("--split_names", nargs='+', type=str, help="Splits available: emea, medline")
    parser.add_argument("--epoch", type=int, help="Number of epoch for the training")
    parser.add_argument("--lr", type=float, help="Learning rate")

    return parser.parse_args()


def load_config_as_namespace(config_file):
    with open(config_file, "r") as f:
        config_dict = yaml.safe_load(f)
    return argparse.Namespace(**config_dict)


if __name__ == "__main__":
    args = parse_arguments()
    config = load_config_as_namespace(args.config)

    dataset_config = list(config.dataset.values())
    model_config = config.model
    trainer_config = list(config.trainer.values())
    
    checkpoint = model_config["checkpoint"]

    training_args = TrainingArguments(
        output_dir=trainer_config[0],
        per_device_train_batch_size=trainer_config[1],
        per_device_eval_batch_size=trainer_config[2],
        num_train_epochs=trainer_config[3],
        learning_rate=trainer_config[4],
        weight_decay=trainer_config[5],
        warmup_steps=trainer_config[6],
        logging_steps=trainer_config[7],
        save_strategy=trainer_config[8],
        save_total_limit=trainer_config[9],
        report_to=trainer_config[10],
    )

    print(training_args)
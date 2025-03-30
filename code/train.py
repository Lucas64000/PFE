import argparse
import yaml
from data.quaero_dataset import DrBenchmarkQUAERO
from model.model_utils import load_hf_model, load_from_dir
from tokenizer.tokenizer_utils import load_tokenizer
from training.trainer import ModelTrainer
from transformers import TrainingArguments
import os 


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml", help="Config path")
    parser.add_argument("--split_names", nargs='+', type=str, help="Splits available: emea, medline")
    parser.add_argument("--epoch", type=int, help="Number of epoch for the training")
    parser.add_argument("--evaluate", action="store_true", help="Set to True to evaluate the model after training") 
    parser.add_argument("--from_dir", action="store_true", help="Load the model from a local directory instead of a Hugging Face checkpoint")

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
    evaluator_config = list(config.evaluator.values())

    checkpoint = model_config["checkpoint"]

    model_name = checkpoint.split("/")[1]

    dataset_path = dataset_config[4]

    dataset = DrBenchmarkQUAERO(
        split_names=dataset_config[0],
        min_len=dataset_config[1],
        test_size=dataset_config[2],
        filepath=dataset_config[3],
    )

    if not os.path.exists(dataset_path):
        print("Existe pas")
        dataset.save_eval_dataset(path=dataset_path)
        print("Dataset d'évaluation sauvegardé")

    if args.from_dir:
        model = load_from_dir(checkpoint)
        base_path = checkpoint.split("/checkpoint-")[0]
        tokenizer = load_tokenizer(base_path)
    else:
        model = load_hf_model(checkpoint, dataset.get_entities())
        tokenizer = load_tokenizer(checkpoint)


    training_args = TrainingArguments(
        output_dir=trainer_config[0],
        logging_dir=trainer_config[1],
        per_device_train_batch_size=trainer_config[2],
        per_device_eval_batch_size=trainer_config[3],
        num_train_epochs=int(trainer_config[4]),
        learning_rate=float(trainer_config[5]),
        weight_decay=trainer_config[6],
        warmup_steps=trainer_config[7],
        logging_steps=trainer_config[8],
        save_strategy=trainer_config[9],
        eval_strategy=trainer_config[9],
        save_total_limit=trainer_config[10],
        report_to=trainer_config[11],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )


    trainer = ModelTrainer(
        model=model,
        dataset=dataset,
        tokenizer=tokenizer,
        training_args=training_args,
        output_dir=trainer_config[0],
    )

    print("Entraînement du modèle en cours")
    trainer.train()
    print("Enregistrement du modèle")
    print("Modèle enregistré dans", trainer.output_dir)

    if args.evaluate:
        print("Évaluation du modèle")
        trainer.evaluate()
import torch
from transformers import Trainer, TrainingArguments
import evaluate
from transformers import DataCollatorForTokenClassification
from tokenizer.tokenizer_utils import tokenize_dataset
import json
import numpy as np

class ModelTrainer:
    def __init__(self, model, dataset, tokenizer, training_args, output_dir="models/"):
        self.model = model
        self.label_list = model.config.id2label
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.training_args = training_args  
        self.output_dir = output_dir
        
        self.metric = evaluate.load("seqeval")
        
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        tokenized_data = tokenize_dataset(self.dataset, self.tokenizer)
        self.train_dataset = tokenized_data.get("train")
        self.eval_dataset = tokenized_data.get("validation")
    
    
    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
    
        results = self.metric.compute(predictions=true_predictions, references=true_labels)
    
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    
    def train(self):
        resume_from_checkpoint = (self.model.config._name_or_path.split("/")[0] == "models")
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(self.output_dir)

    def save_model(self, safe_serialization=False):
        self.model.save_pretrained(self.output_dir, safe_serialization=safe_serialization)
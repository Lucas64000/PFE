import torch
from transformers import Trainer
import evaluate
from transformers import DataCollatorForTokenClassification
from tokenizer.tokenizer_utils import tokenize_dataset
import json
import numpy as np
from plots.plot_utils import plot_confusion_matrix, plot_sklearn_report
import os

class ModelTrainer:
    def __init__(self, model, dataset, tokenizer, training_args):
        training_args.do_train = True

        self.model = model
        self.label_list = model.config.id2label
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.training_args = training_args  
        self.output_dir = training_args.output_dir
        
        self.metric = evaluate.load("seqeval")
        
        self.data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        tokenized_data = tokenize_dataset(self.dataset.data, self.tokenizer)
        self.train_dataset = tokenized_data.get("train")
        self.eval_dataset = tokenized_data.get("validation")

        self._init_trainer()


    def _init_trainer(self):
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.data_collator,
            compute_metrics=self._compute_metrics,
        )
        self.trainer = trainer


    def _compute_metrics(self, p):
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
        resume_from_checkpoint = any(
            os.path.isdir(os.path.join(self.output_dir, d)) and d.startswith("checkpoint-")
            for d in os.listdir(self.output_dir)
        ) if os.path.exists(self.output_dir) else False

        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.tokenizer.save_pretrained(self.output_dir)

    def save_model(self, safe_serialization=False):
        self.model.save_pretrained(self.output_dir, safe_serialization=safe_serialization)
        self.tokenizer.save_pretrained(self.output_dir)
    
    def _predict(self, batch=None):
        if batch is None:
            preds = self.trainer.predict(self.eval_dataset)
            logits = preds.predictions
            labels = preds.label_ids
        else:
            with torch.no_grad():
                out = self.model(**batch)
            logits = out.logits.cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            
        y_pred = np.argmax(logits, axis=-1)
        y_pred = np.concatenate(y_pred).flatten()
        y_true = np.concatenate(labels).flatten()
        
        return (y_pred, y_true)
    
    def evaluate(self, normalize=False):
        y_pred, y_true = self._predict()
        id2label = self.model.config.id2label
        plot_confusion_matrix(y_pred=y_pred, y_true=y_true, id2label=id2label, normalize=normalize)
        plot_sklearn_report(y_pred=y_pred, y_true=y_true, id2label=id2label)
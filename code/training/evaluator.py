from transformers import Trainer
import evaluate
import numpy as np
from plots.plot_utils import plot_confusion_matrix, plot_sklearn_report
from tokenizer.tokenizer_utils import tokenize_dataset

class ModelEvaluator:
    def __init__(self, model, dataset, tokenizer, output_dir="models/"):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.label_list = model.config.id2label
        self.metric = evaluate.load("seqeval")
        self.eval_dataset = tokenize_dataset(dataset, tokenizer)

        self._init_trainer()

    def _init_trainer(self):
        self.trainer = Trainer(
            model=self.model,
            eval_dataset=self.eval_dataset,
            compute_metrics=self._compute_metrics,
        )

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

    def _predict(self):
        preds = self.trainer.predict(self.eval_dataset)
        logits = preds.predictions
        labels = preds.label_ids
            
        y_pred = np.argmax(logits, axis=-1)
        y_pred = np.concatenate(y_pred).flatten()
        y_true = np.concatenate(labels).flatten()

        return y_pred, y_true

    def evaluate(self, normalize=False):
        y_pred, y_true = self._predict()

        plot_confusion_matrix(y_pred=y_pred, y_true=y_true, id2label=self.label_list, normalize=normalize)

        plot_sklearn_report(y_pred=y_pred, y_true=y_true, id2label=self.label_list)
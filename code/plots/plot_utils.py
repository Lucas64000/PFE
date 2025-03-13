import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_classes_repartition(dataset, split):
    label_list = dataset.get_entities()
    df = pd.DataFrame(dataset.data[split])
    ner_tags = [tag for tags in df["ner_tags"] for tag in tags]
    tag_counts = pd.Series(ner_tags).value_counts()
    
    plt.figure(figsize=(18, 6))
    sns.barplot(x=label_list, y=tag_counts, palette="Blues_r")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("NER labels")
    plt.ylabel("Number of occurences")
    plt.title(f"Class distribution in the {split} split")
    plt.show()


def plot_confusion_matrix(trainer):
    y_pred, y_true = trainer._predict()
    id2label = trainer.model.config.id2label
    
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(id2label.keys()))

    plt.figure(figsize=(20, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=list(id2label.values()), yticklabels=list(id2label.values()))

    plt.xlabel("Predicted Labels")
    plt.ylabel("Correct Labels")
    plt.title("Confusion Matrix")
    plt.xticks(rotation=45, ha="right")

    plt.show()


def plot_sklearn_report(trainer):
    y_pred, y_true = trainer._predict()
    id2label = trainer.model.config.id2label

    valid_indices = np.array(y_true) != -100
    y_true_filtered = np.array(y_true)[valid_indices]
    y_pred_filtered = np.array(y_pred)[valid_indices]

    unique_labels = np.unique(y_true_filtered)
    target_names = [id2label[label] for label in unique_labels]

    report_dict = classification_report(
        y_true_filtered,
        y_pred_filtered,
        labels=unique_labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True
    )

    correct_counts = (y_true_filtered == y_pred_filtered).astype(int)
    correct_per_label = {id2label[label]: int(sum(correct_counts[y_true_filtered == label])) for label in unique_labels}

    for label in target_names:
        if label in correct_per_label:
            report_dict[label]["correct"] = f"{correct_per_label[label]}"

    df_report = pd.DataFrame(report_dict).transpose()
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_report.iloc[:-3, :-1], annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Classification Report")
    plt.show()

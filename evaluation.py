import argparse
import os
import numpy as np
import torch

from datasets import ClassLabel, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)


def load_data(data_path, data_generator):
    texts, labels, label_names = [], [], []
    for idx, emotion_dir in enumerate(sorted(os.listdir(data_path))):
        emotion_path = os.path.join(data_path, emotion_dir, data_generator)
        label_names.append(emotion_dir)
        filename = f"{emotion_dir}.txt"
        file_path = os.path.join(emotion_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            texts += [line.strip() for line in lines if line.strip()]
            labels += [idx] * len(lines)
    return texts, labels, label_names


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding='max_length', truncation=True, max_length=256)


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true=p.label_ids,
        y_pred=preds,
        average='weighted',
        zero_division=0,
    )
    accuracy = accuracy_score(p.label_ids, preds)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", required=True, help="Name of the trained model directory (without 'results_' prefix)")
    parser.add_argument("-t", "--dataset_type", required=True, help="Type of dataset (train/test")
    args = parser.parse_args()

    MODEL_PATH = f"./results_{args.model_name}"
    DATASET_TYPE = args.dataset_type
    DATA_PATH = "./data/unrefined"
    DATA_GENERATOR = 'DeepSeek-V3'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts, labels, label_names = load_data(DATA_PATH, DATA_GENERATOR)

    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    dataset = dataset.cast_column("label", ClassLabel(num_classes=len(label_names), names=label_names))

    dataset = dataset.train_test_split(test_size=0.1, stratify_by_column='label', shuffle=True, seed=42)
    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(label_names)).to(device)
    model.eval()

    if DATASET_TYPE in ("train", "test"):
        if DATASET_TYPE == "train":
            tokenized_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        elif DATASET_TYPE == "test":
            tokenized_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

        trainer = Trainer(model=model)

        pred_output = trainer.predict(tokenized_dataset)
        metrics = compute_metrics(pred_output)

        print("Evaluation on test dataset:", metrics)

    else:
        print("Allowed dataset's types are train or test")

import argparse
import numpy as np
import os
import torch

from datasets import ClassLabel, Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)


def load_data(data_path):
    texts, labels, label_names = [], [], []
    for idx, emotion_dir in enumerate(sorted(os.listdir(data_path))):
        emotion_path = os.path.join(data_path, emotion_dir, 'gpt-4o')
        label_names.append(emotion_dir)
        filename = f"{emotion_dir}.txt"
        file_path = os.path.join(emotion_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            texts += [line.strip() for line in lines if line.strip()]
            labels += [idx] * len(lines)
    return texts, labels, label_names


def tokenize_function(examples):
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


class DynamicEvalTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_subset = self.train_dataset.shuffle(seed=np.random.randint(0, 10000)).select(
            range(int(0.1 * len(self.train_dataset)))
        )
        return super().evaluate(eval_dataset=eval_subset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)


def predict_emotion(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)
    pred_label = outputs.logits.argmax(dim=-1).item()
    return label_names[pred_label]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', required=True, help='Name for the trained model directory')
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()

    MODEL_NAME = 'allegro/herbert-base-cased'
    RESULTS_DIR = f"./results_{args.model_name}"
    DATA_PATH = './data/unrefined'
    # DATA_PATH = './data/unrefined_reduced'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    texts, labels, label_names = load_data(DATA_PATH)

    dataset = Dataset.from_dict({'text': texts, 'label': labels})
    dataset = dataset.cast_column("label", ClassLabel(num_classes=len(label_names), names=label_names))
    dataset = dataset.train_test_split(test_size=0.1, stratify_by_column='label', shuffle=True, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_train_dataset = dataset["train"].map(tokenize_function, batched=True)
    tokenized_test_dataset = dataset["test"].map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_names),
        ignore_mismatched_sizes=True
    ).to(device)

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=20,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir='./logs',
        fp16=torch.cuda.is_available(),
        lr_scheduler_type="linear",
        warmup_steps=100,
        logging_strategy="epoch",
        seed=42,
        report_to="none"
    )

    trainer = DynamicEvalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    tokenizer.save_pretrained(RESULTS_DIR)

    trainer.train()

    trainer.save_model(RESULTS_DIR)

    final_metrics = trainer.evaluate(tokenized_test_dataset)
    print("Final evaluation on test dataset:", final_metrics)

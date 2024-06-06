import torch
import numpy as np
from transformers import BertTokenizerFast, BertForTokenClassification, Trainer, TrainingArguments
from datasets import load_metric

labels = ["O", "B-BRAND-safe", "I-BRAND-safe", "B-BRAND-unsafe", "I-BRAND-unsafe", "B-GENERIC", "I-GENERIC"]
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label[word_idx] != 0 else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def create_model():
    model_name = "bert-base-multilingual-cased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name, num_labels=len(labels))
    return model, tokenizer

def train_model(model, tokenizer, train_dataset, val_dataset, training_args):
    train_dataset = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)

    metric = load_metric("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_labels = [[id_to_label[label] for label in label_row] for label_row in labels]
        true_predictions = [
            [id_to_label[pred] for (pred, label) in zip(pred_row, label_row) if label != -100]
            for pred_row, label_row in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

def predict(model, tokenizer, sentence):
    tokens = tokenizer(sentence, return_tensors="pt", truncation=True, is_split_into_words=True)
    with torch.no_grad():
        output = model(**tokens)
    predictions = torch.argmax(output.logits, dim=2)
    predicted_labels = [id_to_label[pred] for pred in predictions[0].tolist()]
    return list(zip(tokenizer.tokenize(sentence), predicted_labels))

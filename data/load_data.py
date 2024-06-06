import json
from datasets import Dataset
from transformers import BertTokenizerFast

labels_mapping = {
    "safe": "B-BRAND-safe",
    "unsafe": "B-BRAND-unsafe",
    "generic": "B-GENERIC"
}

def load_data(file_path='data/data.json', tokenizer=None):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    train_data = data['train']
    val_data = data['validation']

    train_dataset = process_data(train_data, tokenizer)
    val_dataset = process_data(val_data, tokenizer)
    
    return train_dataset, val_dataset

def process_data(data, tokenizer):
    sentences = [item["sentence"] for item in data]
    entities = [item["entities"] for item in data]
    
    tokenized_inputs = tokenizer(sentences, padding='max_length', truncation=True, return_tensors="pt", is_split_into_words=False)
    aligned_labels = align_labels(sentences, entities, tokenized_inputs, tokenizer)
    
    dataset = Dataset.from_dict({
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": aligned_labels
    })
    
    return dataset

def align_labels(sentences, entities, tokenized_inputs, tokenizer):
    aligned_labels = []
    
    for i, sentence in enumerate(sentences):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        labels = ["O"] * len(word_ids)
        sentence_entities = entities[i]
        
        for entity in sentence_entities:
            start_idx = entity['start']
            end_idx = entity['end']
            label = labels_mapping[entity['label']]
            
            char_to_word_mapping = tokenized_inputs.char_to_word_ids(i)
            
            start_word_idx = char_to_word_mapping[start_idx]
            end_word_idx = char_to_word_mapping[end_idx - 1]
            
            if start_word_idx is not None and end_word_idx is not None:
                labels[start_word_idx] = label
                for j in range(start_word_idx + 1, end_word_idx + 1):
                    labels[j] = label.replace('B-', 'I-')
        
        label_ids = [label_to_id[label] if label != "O" else -100 for label in labels]
        aligned_labels.append(label_ids)
    
    return aligned_labels

label_to_id = {
    "O": 0,
    "B-BRAND-safe": 1,
    "I-BRAND-safe": 2,
    "B-BRAND-unsafe": 3,
    "I-BRAND-unsafe": 4,
    "B-GENERIC": 5,
    "I-GENERIC": 6
}

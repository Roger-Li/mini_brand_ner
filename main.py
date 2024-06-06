# export KMP_DUPLICATE_LIB_OK=TRUE

import argparse
from data.load_data import load_data
from models.ner_model import create_model, train_model, predict
from utils.utils import print_predictions
from transformers import TrainingArguments, BertTokenizerFast

def main():
    parser = argparse.ArgumentParser(description="Train a Named Entity Recognition model.")
    
    # Add arguments
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save the model checkpoints and logs.')
    parser.add_argument('--evaluation_strategy', type=str, default='epoch', choices=['no', 'steps', 'epoch'], help='Evaluation strategy.')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the optimizer.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=16, help='Batch size for evaluation.')
    parser.add_argument('--num_train_epochs', type=int, default=3, help='Number of epochs to train the model.')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for the optimizer.')

    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = create_model()

    # Load data
    train_dataset, val_dataset = load_data(tokenizer=tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy=args.evaluation_strategy,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
    )

    # Train model
    train_model(model, tokenizer, train_dataset, val_dataset, training_args)

    # Predict example
    sentence = "Phone case compatible with Apple"
    predictions = predict(model, tokenizer, sentence)
    print_predictions(sentence, predictions)

if __name__ == "__main__":
    main()

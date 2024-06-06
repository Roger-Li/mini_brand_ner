def print_predictions(sentence, predictions):
    print(f"Sentence: {sentence}")
    for token, label in predictions:
        print(f"{token}: {label}")

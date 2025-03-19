import argparse
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer


def predict_emotion(model, sentence):
    label_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    pred_label = outputs.logits.argmax(dim=-1).item()
    print("outputs: ", outputs)
    print("emotion: ", pred_label)
    return label_names[pred_label]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', required=True, help='Name for the trained model directory')
    args = parser.parse_args()

    MODEL_PATH = f"./results_{args.model_name}"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    print("Hi! Enter your sentence to classify emotion.")
    print("Enter 'exit' or 'quit' to stop program.\n")

    while True:
        user_input = input("Your sentence: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Program has been stopped.")
            break

        predict_emotion(model, user_input)

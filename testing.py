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
    print("pred_label: ", pred_label)
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

    example_sentence = "Już dość mam twojego ciągłego narzekania i negatywnego podejścia!"
    print(f'Zdanie: "{example_sentence}" => Emocja: {predict_emotion(model, example_sentence)}')

    example_sentence = "Brudne naczynia zalegały w zlewie od kilku tygodni."
    print(f'Zdanie: "{example_sentence}" => Emocja: {predict_emotion(model, example_sentence)}')

    example_sentence = "Ciemność zasnuła pokój, a cisza była przerażająca."
    print(f'Zdanie: "{example_sentence}" => Emocja: {predict_emotion(model, example_sentence)}')

    example_sentence = "Nasz nowy projekt artystyczny okazał się być wielkim sukcesem."
    print(f'Zdanie: "{example_sentence}" => Emocja: {predict_emotion(model, example_sentence)}')

    example_sentence = "Na lokalnym targu można znaleźć świeże warzywa i owoce od rolników."
    print(f'Zdanie: "{example_sentence}" => Emocja: {predict_emotion(model, example_sentence)}')

    example_sentence = "Niegdyś zielone drzewa teraz stoją wiotkie, straciły swój blask."
    print(f'Zdanie: "{example_sentence}" => Emocja: {predict_emotion(model, example_sentence)}')

    example_sentence = "Wakacje niespodziewanie przedłużyły się o tydzień dzięki nadprogramowym dniom wolnym."
    print(f'Zdanie: "{example_sentence}" => Emocja: {predict_emotion(model, example_sentence)}')

    example_sentence = "Jestem zdziwiony twoim niespotykanym talentem."
    print(f'Zdanie: "{example_sentence}" => Emocja: {predict_emotion(model, example_sentence)}')

import argparse
import os
import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DOMAINS = [
    "Praca",
    "Film",
    "Rozrywka",
    "Miłość",
    "Rodzina",
    "Edukacja",
    "Sport",
    "Zdrowie",
    "Technologia",
    "Moda",
    "Podróże",
    "Kultura",
    "Religia",
    "Polityka",
    "Finanse",
    "Przyjaźń",
    "Gotowanie",
    "Sztuka",
    "Muzyka",
    "Literatura",
    "Ekologia",
    "Motoryzacja",
    "Nauka",
    "Zwierzęta",
    "Przestępczość",
    "Gry komputerowe",
    "Żałoba",
    "Wychowanie dzieci",
    "Sfera niecenzuralna",
    "Biznes",
    "Architektura",
    "Przedsiębiorczość",
    "Samorozwój",
    "Prawo",
    "Języki obce",
    "Historia",
    "Seksualność",
    "Internet",
    "Social media",
    "Wolontariat",
    "Dieta i odżywianie",
    "Uroda",
    "Produkcja",
    "Medytacja",
    "Ogrodnictwo",
    "Rolnictwo",
    "Emigracja",
    "Marzenia i cele życiowe",
    "Emocje i uczucia",
]


def query_local_bielik(model, tokenizer, device, prompt, temperature=1.0, max_new_tokens=10000):
    messages = [
        {
            "role": "system",
            "content": (
                "1. Odpowiadaj wyłącznie w języku polskim. "
                "2. Każde zdanie ma być w nowej linii i bez numeratorów. "
                "3. Nie używaj słów pochodnych od słowa 'Twoja'. "
                "4. W odpowiedzi nie może być twoich komentarzy, tylko same zdania."
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


def save_response_to_file(response, emotion, model_name, words):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = os.path.join("data", "raw", emotion, model_name)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{emotion}_{words}_{timestamp}.txt")

    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(response + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', required=False, help='Number of iterations', default=1)
    parser.add_argument('-t', '--temperature', required=False, help='Temperature of model', default=1.0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "speakleash/Bielik-11B-v2.3-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)

    local_model_label = "Bielik-11B-v2.3-Instruct"
    temperature = float(args.temperature)
    max_new_tokens = 20000
    num_sentences = 25
    iterations = int(args.iterations)
    words_min = 1
    words_max = 12
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    for i in range(iterations):
        for words in range(words_min, words_max + 1):
            for emotion in emotions:
                for domain in DOMAINS:
                    prompt = (
                        f"Napisz {num_sentences} kolejnych unikalnych różnorodnych zdań dotyczących "
                        f"różnych aspektów z obszaru {domain} składających się dokładnie z {words} słów, "
                        f"które można zaklasyfikować do klasy emocji {emotion}."
                    )

                    try:
                        response = query_local_bielik(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            prompt=prompt,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens
                        )
                    except Exception as e:
                        response = query_local_bielik(
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            prompt=prompt,
                            temperature=temperature,
                            max_new_tokens=max_new_tokens
                        )

                    save_response_to_file(response, emotion, local_model_label, words)

                    print(f"Generated sentences {emotion} - {words}/{words_max} words for {domain} domain: "
                          f"iteration {i + 1}/{iterations}")


if __name__ == "__main__":
    main()

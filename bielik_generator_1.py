import argparse
import os
import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def query_local_bielik(model, tokenizer, device, prompt, temperature=1.0, max_new_tokens=200):
    """
    Generuje tekst na podstawie podanego promptu przy użyciu lokalnego modelu Bielik-11B-v2.3-Instruct.
    """

    # Przykładowe komunikaty w stylu "chat" (możesz je dostosować)
    messages = [
        {
            "role": "system",
            "content": "Odpowiadaj wyłącznie w języku polskim. Każde zdanie ma być w nowej linii i bez numeratorów"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    # Dla modelu Bielik-11B-v2.3-Instruct używamy wbudowanej metody apply_chat_template
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # Generowanie
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id  # Na wypadek braku tokenu do pad
        )

    # Dekodowanie na czytelny tekst
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text


def save_response_to_file(response, emotion, model_name):
    """
    Zapisuje wygenerowany tekst do pliku w katalogu zgodnym z danym emotion i modelem.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = os.path.join("data", "raw", emotion, model_name)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{emotion}_{timestamp}.txt")

    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(response + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', required=False, help='Number of iterations', default=1)
    args = parser.parse_args()
    # --------------------------
    # KONFIGURACJA MODELU BIELIK
    # --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "speakleash/Bielik-11B-v2.3-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.to(device)

    # --------------------------
    # USTAWIENIA GENERACJI
    # --------------------------
    local_model_label = "Bielik-11B-v2.3-Instruct"  # używamy tego w nazwach folderów
    temperature = 1.0
    max_new_tokens = 200  # maksymalna długość wygenerowanego fragmentu
    num_sentences = 50  # ile zdań ma zawierać jeden "prompt"
    iterations = args.iterations  # ile razy powtarzamy generację dla danej emocji
    words_min = 1
    words_max = 12
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    # --------------------------
    # PĘTLA GENERUJĄCA TEKSTY
    # --------------------------
    for i in range(iterations):
        for w in range(words_min, words_max + 1):
            for emotion in emotions:
                # PROMPT w stylu data_generator.py
                prompt = (
                    f"Napisz kolejne nowe inne unikalne zdania (tylko zdanie! bez numeratorów, tylko od nowej linii) "
                    f"od {w} do {words_max} słów, które można zaklasyfikować do klasy emocji {emotion}. "
                    f"({num_sentences} przykładów) - dodaj jeszcze większą różnorodność, ma dotyczyć różnych aspektów "
                    "życia (pomiń słowa pochodne od \"twoja\") oraz staraj się nie zaczynać zdań od przyimków i "
                    "spójników - Ale pamiętaj ma nie być numeratorów! Dodatkowo nie chcę pokazanego dodatkowego "
                    "zwracanego \"reasoning\"!"
                )

                # Generujemy tekst lokalnie przez Bielik
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
                    # Druga próba (jeżeli wystąpił jakiś chwilowy błąd)
                    response = query_local_bielik(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        temperature=temperature,
                        max_new_tokens=max_new_tokens
                    )

                # Zapisujemy dane do pliku
                save_response_to_file(response, emotion, local_model_label)

                # Logujemy postęp
                print(f"Generated sentences ({w} - {words_max} words) for '{emotion}': iteration {i + 1}/{iterations}")


if __name__ == "__main__":
    main()

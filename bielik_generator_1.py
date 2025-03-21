import argparse
import os
import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def query_local_bielik(model, tokenizer, device, prompt, temperature=1.0, max_new_tokens=10000):
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

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    inputs = inputs.to(device)
    model.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return generated_text


def save_response_to_file(response, emotion, model_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = os.path.join("data", "raw", emotion, model_name)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{emotion}_{timestamp}.txt")

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
    temperature = args.temperature
    max_new_tokens = 200
    num_sentences = 50
    iterations = args.iterations
    words_min = 1
    words_max = 12
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    for i in range(iterations):
        for w in range(words_min, words_max + 1):
            for emotion in emotions:
                prompt = (
                    f"Napisz kolejne nowe inne unikalne zdania (tylko zdanie! bez numeratorów, tylko od nowej linii) "
                    f"od {w} do {words_max} słów, które można zaklasyfikować do klasy emocji {emotion}. "
                    f"({num_sentences} przykładów) - dodaj jeszcze większą różnorodność, ma dotyczyć różnych aspektów "
                    "życia (pomiń słowa pochodne od \"twoja\") oraz staraj się nie zaczynać zdań od przyimków i "
                    "spójników - Ale pamiętaj ma nie być numeratorów! Dodatkowo nie chcę pokazanego dodatkowego "
                    "zwracanego \"reasoning\"!"
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

                save_response_to_file(response, emotion, local_model_label)

                print(f"Generated sentences ({w} - {words_max} words) for '{emotion}': iteration {i + 1}/{iterations}")


if __name__ == "__main__":
    main()

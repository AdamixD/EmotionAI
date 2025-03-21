import time

import openai
import os
import datetime


def query_gpt(api_key, model, prompt, temperature):
    openai.api_key = api_key
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message['content']


def save_response_to_file(response, emotion, model):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = f"/Users/adamdabkowski/PycharmProjects/MasterThesis/data/raw/{emotion}/{model}/"
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{emotion}_{timestamp}.txt")

    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(response + "\n")


def main(api_key, model, temperature, num_sentences, iterations, emotions):
    for i in range(iterations):
        for emotion in emotions:
            prompt = (f"Napisz kolejne nowe inne unikalne zdania (tylko zdanie! bez numeratorów, tylko od nowej linii) "
                      f"od 10 do 12 słów, które można zaklasyfikować do klasy emocji {emotion}. ({num_sentences} przykładów) - dodaj jeszcze "
                      f"większą różnorodność, ma dotyczyć różnych aspektów życia (pomiń słowa pochodne od \"twoja\") "
                      f"oraz staraj się nie zaczynać zdań od przyimków i spójników - Ale pamiętaj ma nie być numeratorów! "
                      f"Dodatkowo nie chcę pokazanego dodatkowego zwracanego \"reasoning\"!")

            try:
                response = query_gpt(api_key, model, prompt, temperature)
            except Exception as e:
                response = query_gpt(api_key, model, prompt, temperature)

            save_response_to_file(response, emotion, model)
            print(f"Generated sentences ({emotion}): {(i + 1) * num_sentences}")
            time.sleep(10)


if __name__ == "__main__":
    API_KEY = ""
    MODEL = "o3-mini"
    # MODEL = "gpt-4o"
    TEMPERATURE = 1.0  # (0.0 - 1.5)
    NUM_SENTENCES = 50
    ITERATIONS = 20
    EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    main(API_KEY, MODEL, TEMPERATURE, NUM_SENTENCES, ITERATIONS, EMOTIONS)

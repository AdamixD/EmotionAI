import argparse
import datetime
import os
import time

from openai import OpenAI

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_MAPPING = {
    "v3": "deepseek-chat",
    "r1": "deepseek-reasoner"
}


class DeepSeekClient:
    def __init__(self, api_key, model_version="v3"):
        self.client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_BASE_URL
        )
        self.model = MODEL_MAPPING[model_version.lower()]

    def generate(self, prompt, temperature=0.8, frequency_penalty=0.5, top_p=0.9, max_tokens=4000):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "1. Odpowiadaj wyłącznie w języku polskim\n"
                            "2. Formatuj odpowiedź jako lista zdań\n"
                            "3. Każde zdanie w nowej linii bez numeracji\n"
                            "4. Unikaj słów: 'Twoja', 'Twój', 'Twoje' i podobnych\n"
                            "5. Wygeneruj dokładnie 50 zdań spełniających kryteria.\n"
                            "6. Słowa nie we wszytkich zdaniach mają być unikalne lub zastąpione synonimem.\n"
                            "7. Nie dodawaj żadnych komentarzy, zwrócone mają być tylko zdania"
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=frequency_penalty,
                top_p=top_p,
                stream=False
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            raise RuntimeError(f"API Error: {str(e)}")


def save_response(response, emotion, model_name, words):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = os.path.join("data", "raw", emotion, model_name)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{emotion}_{words}_{timestamp}.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True, help="Klucz API DeepSeek")
    parser.add_argument("--model", choices=["v3", "r1"], default="v3", help="Wersja modelu (v3/r1)")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    args.temperature = max(0.0, min(args.temperature, 2))

    api_client = DeepSeekClient(args.api_key, args.model)
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    for i in range(args.iterations):
        for words in range(1, 13):
            for emotion in emotions:
                prompt = (
                    f"Wygeneruj 50 unikalnych zdań w języku polskim spełniających kryteria:\n"
                    f"- Temat: różne dziedziny życia ludzi i zwierząt\n"
                    f"- Emocja: zdanie ma być zaklasyfikowane do emocji {emotion.capitalize()}\n"
                    f"- Długość: {words}-{words + 2} słów\n"
                    f"- Format: każde zdanie w nowej linii\n"
                )

                response = api_client.generate(
                    prompt=prompt,
                    temperature=args.temperature,
                    frequency_penalty=0.6,
                    top_p=0.9,
                    max_tokens=8000,
                )

                if not response or len(response.split('\n')) < 30:
                    print(f"Otrzymano niepełną odpowiedź dla {emotion}, {words} słów")
                    continue

                save_response(response, emotion, f"DeepSeek-{args.model.upper()}", words)
                print(f"Emocja: {emotion} | Słowa: {words}-{words + 2} | Iteracja: {i + 1}/{args.iterations}")

                time.sleep(1)


if __name__ == "__main__":
    main()

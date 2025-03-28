import argparse
import os
import datetime
import warnings

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def query_deepseek_r1(model, tokenizer, device, prompt, temperature=0.6, max_new_tokens=8000):
    think_sequence = "<think>\n"
    think_tokens = tokenizer.encode(think_sequence, add_special_tokens=False)

    formatted_prompt = (
        f"""
        <|im_start|>user
        {prompt}
        <|im_end|>
        <|im_start|>assistant
        {think_sequence}
        """
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        max_length=4096,
        truncation=True,
        padding=True
    ).to(device)

    generation_config = GenerationConfig(
        temperature=max(0.5, min(temperature, 0.7)),
        top_p=0.9,
        top_k=40,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
        forced_decoder_ids=[(i, tid) for i, tid in enumerate(think_tokens)],
        eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
    )

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )

    response_start = inputs["input_ids"].shape[1] + len(think_tokens)

    response = tokenizer.decode(
        outputs[0][response_start:],
        skip_special_tokens=True
    ).strip()

    return response


def save_response_to_file(response, emotion, model_name, words):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    directory = os.path.join("data", "raw", emotion, model_name)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"{emotion}_{words}_{timestamp}.txt")

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(response + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', type=int, default=1, help='Iterations (default: 1)')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature (0.5-0.7, default: 0.6)')
    args = parser.parse_args()

    args.temperature = max(0.5, min(args.temperature, 0.7))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "deepseek-ai/DeepSeek-R1"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            revision="a06b8b7013d2e0c5b274412c685d467a6c4dc8d0"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            load_in_8bit=False,
            quantization_config=None,
            revision="a06b8b7013d2e0c5b274412c685d467a6c4dc8d0"
        )

    except Exception as e:
        raise RuntimeError(f"Error during loading model {model_name}: {str(e)}")

    local_model_label = "DeepSeek-R1"
    emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    words_range = range(1, 13)

    for i in range(args.iterations):
        for words in words_range:
            for emotion in emotions:
                prompt = (
                    f"""
                    Napisz 50 kolejnych unikalnych zdań w języku polskim dotyczących różnych dziedzin życia 
                    ludzi i zwierząt. Każde zdanie powinno:
                    1. Zawierać od {words} do {words + 2} słów
                    2. Być w nowej linii bez numeracji
                    3. Unikać słów pochodnych od 'Twoja'
                    4. Wyrażać emocję: {emotion.capitalize()}
                    5. Zwrócone mają być tylko i wyłącznie zdania, bez komentarzy modelu
                    """
                )

                try:
                    response = query_deepseek_r1(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_new_tokens=8000,
                    )

                except Exception as e:
                    warnings.warn(f"Error during generation: {str(e)}, another attempt ...")
                    response = query_deepseek_r1(
                        model=model,
                        tokenizer=tokenizer,
                        device=device,
                        prompt=prompt,
                        temperature=args.temperature
                    )

                save_response_to_file(response, emotion, local_model_label, words)
                print(f"Emotion: {emotion} | Words: ({words} - {words + 2}) | Iteration: {i + 1}/{args.iterations}")


if __name__ == "__main__":
    main()

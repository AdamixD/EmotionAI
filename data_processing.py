import os
import re


def clean_text(text):
    cleaned_lines = []
    for line in text.splitlines():
        cleaned_line = re.sub(r'^\s*[\d\[\(]+[\]\)\.\s]*\s*', '', line).strip()  # ('1. ', '23) ', '[4] ', ' ')
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    return '\n'.join(cleaned_lines)


def process_files(base_input_path, base_output_path, model=None):
    for emotion in os.listdir(base_input_path):
        emotion_path = os.path.join(base_input_path, emotion)
        if not os.path.isdir(emotion_path):
            continue

        model_path = os.path.join(emotion_path, model)
        if not os.path.isdir(model_path):
            print(f"No data found for model: {model} in emotion: {emotion}")
            continue

        output_dir = os.path.join(base_output_path, emotion, model)
        os.makedirs(output_dir, exist_ok=True)
        output_file_path = os.path.join(output_dir, f"{emotion}.txt")

        merged_text = []
        for file_name in os.listdir(model_path):
            file_path = os.path.join(model_path, file_name)
            if not file_name.endswith(".txt"):
                continue

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
                cleaned_text = clean_text(text)
                merged_text.append(cleaned_text)

        with open(output_file_path, "w", encoding="utf-8") as output_file:
            output_file.write("\n".join(merged_text))

        line_count = sum(1 for _ in open(output_file_path, "r", encoding="utf-8"))

        print(f"Processed ({emotion}) data ({line_count} lines) for {model} model: {output_file_path}")


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_input_path = os.path.join(project_dir, "EmotionAI", "data", "raw")
    base_output_path = os.path.join(project_dir, "EmotionAI", "data", "unrefined")

    model = "DeepSeek-V3"
    process_files(base_input_path, base_output_path, model)

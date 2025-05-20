import argparse
import datetime as dt
import time

from pathlib import Path
from typing import List, Tuple

from openai import OpenAI


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_MAPPING = {"v3": "deepseek-chat", "r1": "deepseek-reasoner"}
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


class DeepSeekClient:
    def __init__(self, api_key: str, model_version: str = "v3"):
        self.client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        self.model = MODEL_MAPPING[model_version.lower()]

        self._system_msg = {
            "role": "system",
            "content": (
                "Jesteś klasyfikatorem emocji. "
                "Dla KAŻDEJ linijki wejściowej zwróć dokładnie jedną linijkę "
                "w formacie '<emotion>: <identyczne zdanie>'. "
                "Etykieta musi być JEDNYM słowem z listy "
                "angry, disgust, fear, happy, neutral, sad, surprise. "
                "Nie dodawaj numeracji ani dodatkowych znaków. "
                "Jeśli masz odczucie że jest to tylko relacja narratora i wyraźny "
                "opis pewnej sytuacji wówczas sklasyfikuj zdanie jako 'neutral', "
                "jednakże pamiętaj, że nawet czasi relacje narratorskie mogą przejwiać inne emocje."
            ),
        }

    def classify_batch(
        self, sentences: List[str], temperature: float = 0.0
    ) -> List[Tuple[str, str]]:
        if not sentences:
            return []

        user_msg = {"role": "user", "content": "\n".join(sentences)}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[self._system_msg, user_msg],
            temperature=temperature,
            stream=False,
        )

        raw_lines = resp.choices[0].message.content.strip().splitlines()

        # [print(raw_line) for raw_line in raw_lines]
        if len(raw_lines) != len(sentences):
            print(
                f"[WARN] Mismatch: expected {len(sentences)} lines, "
                f"received {len(raw_lines)}."
            )

        parsed = []
        for i, original in enumerate(sentences):
            try:
                ln = raw_lines[i]
            except IndexError:
                parsed.append(("neutral", original))
                continue

            if ": " not in ln:
                parsed.append(("neutral", original))
                continue

            label, sent = ln.split(": ", 1)
            label = label.strip().lower()
            if label not in EMOTIONS:
                label = "neutral"
            parsed.append((label, sent.strip()))

        return parsed


def save_emotion_lines(lines, emotion, model_name, timestamp):
    out_dir = Path("data", "raw", emotion, model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_path = out_dir / f"{emotion}_classified_{timestamp}.txt"
    with file_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def process_txt_file(path: Path, client: DeepSeekClient, pause: float):
    with path.open("r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    if not lines:
        print(f"[SKIP] Empty file: {path}")
        return

    buckets = {e: [] for e in EMOTIONS}

    for batch in chunks(lines, 50):
        for label, sentence in client.classify_batch(batch):
            buckets[label].append(sentence)
        time.sleep(pause)

    timestamp = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = f"DeepSeek-{client.model.split('-')[-1].upper()}"
    created = 0
    for emo, lst in buckets.items():
        if lst:
            save_emotion_lines(lst, emo, model_name, timestamp)
            created += 1

    print(f"[INFO] {path} → {created}/7 files, {len(lines)} sentences")


def main():
    ap = argparse.ArgumentParser("Batch 50× '{emotion}: sentence' classifier")
    ap.add_argument("--api-key", required=True, help="Klucz API DeepSeek")
    ap.add_argument("--model", choices=["v3", "r1"], default="v3")
    ap.add_argument("--sleep", type=float, default=0.5, help="Pause between batches (s)")
    args = ap.parse_args()

    client = DeepSeekClient(args.api_key, args.model)
    txt_files = sorted(Path("data_web_scrapping/data/unrefined").rglob("*.txt"))
    if not txt_files:
        print("No files .txt in data_web_scrapping/data/unrefined")
        return

    t0 = time.time()
    for fp in txt_files:
        process_txt_file(fp, client, args.sleep)

    h, rem = divmod(int(time.time() - t0), 3600)
    m, s = divmod(rem, 60)
    print(f"[FINISH] - {h:02d}h {m:02d}m {s:02d}s")


if __name__ == "__main__":
    main()

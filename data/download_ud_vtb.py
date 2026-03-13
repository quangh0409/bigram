from pathlib import Path
from datasets import load_dataset


def extract_sentences_from_wikipedia(dataset) -> list[str]:
    sentences: list[str] = []
    for item in dataset:
        if "text" in item:
            text = item["text"].strip()
            if text:
                for line in text.split("\n"):
                    line = line.strip()
                    if line and len(line) > 10:
                        sentences.append(line)
    return sentences


def download_and_build_corpus(output_path: Path, max_samples: int = 10000) -> int:
    print("Loading Wikipedia Vietnamese dataset from Hugging Face...")
    dataset = load_dataset("tdtunlp/wikipedia_vi", split="train")

    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))

    print(f"Processing {len(dataset)} Wikipedia articles...")
    all_sentences = extract_sentences_from_wikipedia(dataset)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(all_sentences), encoding="utf-8")
    return len(all_sentences)


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "vi_sentences_wikipedia.txt"
    count = download_and_build_corpus(output, max_samples=10000)
    print(f"Saved {count} sentences to {output}")

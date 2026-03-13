import argparse
import random
import sys
from pathlib import Path

from ngram_vi import BaseNgramVietnameseLM
from data.download_ud_vtb import download_and_build_corpus

# Fix encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


CORPUS_PATH = Path("data/vi_sentences_wikipedia.txt")


def ensure_corpus(max_samples: int = 10000) -> list[str]:
    if not CORPUS_PATH.exists():
        count = download_and_build_corpus(CORPUS_PATH, max_samples=max_samples)
        print(f"Downloaded corpus with {count} sentences")

    sentences = [
        line.strip()
        for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sentences


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Vietnamese n-gram language model and run demo.",
    )
    parser.add_argument(
        "--n",
        type=int,
        choices=[2, 4],
        default=2,
        help="N-gram order to use.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Laplace smoothing coefficient.",
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=10,
        help="Number of sentences to generate.",
    )
    parser.add_argument(
        "--min-len",
        type=int,
        default=5,
        help="Minimum generated sentence length in tokens.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=20,
        help="Maximum generated sentence length in tokens.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of Wikipedia articles to use for training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    sentences = ensure_corpus(max_samples=args.max_samples)

    model = BaseNgramVietnameseLM(n=args.n, alpha=args.alpha)
    model.train(sentences)

    print(f"N-gram order: {args.n}")
    print(f"Laplace alpha: {args.alpha}")
    print(f"Training sentences: {len(sentences)}")
    print(f"Vocabulary size: {len(model.vocab)}")

    test_sentences = [
        "Hôm nay trời đẹp lắm",
        "Tôi thích học xử lý ngôn ngữ tự nhiên",
        "Con mèo đang ngủ trên ghế",
    ]

    print("\nSentence probabilities (with Laplace smoothing):")
    for sent in test_sentences:
        result = model.sentence_probability(sent)
        print(f"- {sent}")
        print(f"  P(sentence)  = {result.probability:.12e}")
        print(f"  log P(sentence) = {result.log_probability:.6f}")

    print("\nGenerated sentences:")
    for i in range(1, args.num_sentences + 1):
        generated = model.generate_sentence(
            min_len=args.min_len,
            max_len=args.max_len,
        )
        print(f"{i:02d}. {generated}")


if __name__ == "__main__":
    main()

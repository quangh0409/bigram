from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)



def tokenize_syllables(sentence: str) -> list[str]:
    return TOKEN_PATTERN.findall(sentence.lower())


@dataclass
class SentenceProbability:
    sentence: str
    log_probability: float
    probability: float


class BaseNgramVietnameseLM:
    def __init__(self, n: int = 2, alpha: float = 1.0) -> None:
        if n < 2:
            raise ValueError("n must be >= 2")
        self.n = n
        self.alpha = alpha
        self.context_counts: Counter[tuple[str, ...]] = Counter()
        self.ngram_counts: DefaultDict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def train(self, sentences: list[str]) -> None:
        for sentence in sentences:
            tokens = ["<s>"] * (self.n - 1) + tokenize_syllables(sentence) + ["</s>"]
            for tok in tokens:
                self.vocab.add(tok)

            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1 : i])
                next_tok = tokens[i]
                self.context_counts[context] += 1
                self.ngram_counts[context][next_tok] += 1

    def conditional_probability(self, context: tuple[str, ...], next_tok: str) -> float:
        v = len(self.vocab)
        numerator = self.ngram_counts[context][next_tok] + self.alpha
        denominator = self.context_counts[context] + self.alpha * v
        if denominator == 0:
            return 1.0 / max(v, 1)
        return numerator / denominator

    def sentence_probability(self, sentence: str) -> SentenceProbability:
        tokens = ["<s>"] * (self.n - 1) + tokenize_syllables(sentence) + ["</s>"]
        log_p = 0.0

        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - self.n + 1 : i])
            p = self.conditional_probability(context, tokens[i])
            log_p += math.log(p)

        return SentenceProbability(
            sentence=sentence,
            log_probability=log_p,
            probability=math.exp(log_p),
        )

    def generate_sentence(
        self,
        max_len: int = 25,
        min_len: int = 1,
        max_attempts: int = 30,
    ) -> str:
        if max_len < 1:
            raise ValueError("max_len must be >= 1")
        if min_len < 0:
            raise ValueError("min_len must be >= 0")
        if min_len > max_len:
            raise ValueError("min_len must be <= max_len")

        best = ""
        for _ in range(max_attempts):
            candidate = self._generate_once(max_len=max_len)
            if len(candidate.split()) >= min_len:
                return candidate
            if len(candidate.split()) > len(best.split()):
                best = candidate

        return best

    def _generate_once(self, max_len: int) -> str:
        history = ["<s>"] * (self.n - 1)
        generated: list[str] = []

        for _ in range(max_len):
            next_tokens, weights = self._next_distribution_with_backoff(history)
            next_tok = random.choices(next_tokens, weights=weights, k=1)[0]

            if next_tok == "</s>":
                break
            if next_tok != "<s>":
                generated.append(next_tok)

            history.append(next_tok)
            history = history[-(self.n - 1) :]

        if not generated:
            return ""
        return " ".join(generated)

    def _next_distribution_with_backoff(
        self,
        history: list[str],
    ) -> tuple[list[str], list[float]]:
        max_order = min(len(history), self.n - 1)
        for order in range(max_order, 0, -1):
            context = tuple(history[-order:])
            if context in self.ngram_counts and self.ngram_counts[context]:
                next_tokens = list(self.ngram_counts[context].keys())
                weights = [self.ngram_counts[context][tok] for tok in next_tokens]
                return next_tokens, weights

        return ["</s>"], [1.0]


class BigramVietnameseLM(BaseNgramVietnameseLM):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__(n=2, alpha=alpha)

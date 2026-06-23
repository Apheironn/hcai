from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .dataset import CLASS_NAMES


SPORTS_WORDS = {"game", "team", "win", "league", "player", "score", "coach", "season", "match", "cup"}
BUSINESS_WORDS = {"market", "stock", "company", "profit", "shares", "bank", "trade", "economy", "investor", "revenue"}
SCI_WORDS = {"software", "computer", "technology", "internet", "microsoft", "chip", "digital", "online", "device", "tech"}
WORLD_WORDS = {"government", "president", "war", "country", "minister", "official", "nation", "peace", "election", "united"}


KEYWORD_MAP = {
    "world": WORLD_WORDS,
    "sports": SPORTS_WORDS,
    "business": BUSINESS_WORDS,
    "scitech": SCI_WORDS,
}


def highlight_keywords(text, max_len=280):
    import html
    import re

    snippet = text if len(text) <= max_len else text[: max_len - 3] + "..."
    escaped = html.escape(snippet)
    for css_class, words in KEYWORD_MAP.items():
        for word in sorted(words, key=len, reverse=True):
            pattern = re.compile(rf"\b({re.escape(word)})\b", re.IGNORECASE)
            escaped = pattern.sub(rf'<span class="kw-{css_class}">\1</span>', escaped)
    return escaped


@dataclass
class ExpertReport:
    name: str
    overall_accuracy: float
    per_class_accuracy: dict
    strengths: str
    weaknesses: str


class SimulatedExpert(ABC):
    name: str

    @abstractmethod
    def predict(self, text: str) -> int:
        pass

    def predict_batch(self, texts):
        return np.array([self.predict(t) for t in texts])

    def analyze(self, dataset) -> ExpertReport:
        preds = self.predict_batch(dataset.test_texts)
        labels = dataset.test_labels
        overall = float(np.mean(preds == labels))

        per_class = {}
        for idx, name in enumerate(CLASS_NAMES):
            mask = labels == idx
            if mask.sum() == 0:
                per_class[name] = 0.0
            else:
                per_class[name] = round(float(np.mean(preds[mask] == labels[mask])), 4)

        ranked = sorted(per_class.items(), key=lambda x: x[1], reverse=True)
        strengths = ", ".join(f"{n} ({a})" for n, a in ranked[:2])
        weaknesses = ", ".join(f"{n} ({a})" for n, a in ranked[-2:])
        return ExpertReport(
            name=self.name,
            overall_accuracy=round(overall, 4),
            per_class_accuracy=per_class,
            strengths=strengths,
            weaknesses=weaknesses,
        )


class TopicKeywordExpert(SimulatedExpert):
    name = "Topic Keyword Expert"

    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)

    def _score(self, text, words):
        tokens = set(text.lower().split())
        return len(tokens & words)

    def predict(self, text: str) -> int:
        scores = {
            0: self._score(text, WORLD_WORDS),
            1: self._score(text, SPORTS_WORDS),
            2: self._score(text, BUSINESS_WORDS),
            3: self._score(text, SCI_WORDS),
        }
        best = max(scores.values())
        if best == 0:
            return int(self.rng.integers(0, 4))
        top = [k for k, v in scores.items() if v == best]
        return int(top[0]) if len(top) == 1 else int(self.rng.choice(top))

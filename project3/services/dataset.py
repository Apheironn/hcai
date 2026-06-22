import numpy as np
import pandas as pd
from datasets import load_dataset

CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
_CACHE = None


def _row_text(row):
    if "text" in row:
        return row["text"]
    return f"{row['title']} {row['description']}"


class AgNewsDataset:
    def __init__(self, train_texts, train_labels, test_texts, test_labels):
        self.train_texts = train_texts
        self.train_labels = train_labels
        self.test_texts = test_texts
        self.test_labels = test_labels

    @classmethod
    def load(cls):
        global _CACHE
        if _CACHE is not None:
            return _CACHE

        raw = load_dataset("fancyzhx/ag_news")
        train_df = raw["train"].to_pandas()
        test_df = raw["test"].to_pandas()

        train_texts = train_df.apply(_row_text, axis=1).tolist()
        test_texts = test_df.apply(_row_text, axis=1).tolist()
        train_labels = train_df["label"].to_numpy(dtype=int)
        test_labels = test_df["label"].to_numpy(dtype=int)

        _CACHE = cls(train_texts, train_labels, test_texts, test_labels)
        return _CACHE

    @property
    def n_train(self):
        return len(self.train_texts)

    @property
    def n_test(self):
        return len(self.test_texts)

    def class_counts(self):
        counts = {name: 0 for name in CLASS_NAMES}
        for label in self.train_labels:
            counts[CLASS_NAMES[label]] += 1
        return counts

    def preview(self, n=5):
        rows = []
        for i in range(min(n, self.n_train)):
            text = self.train_texts[i]
            if len(text) > 120:
                text = text[:117] + "..."
            rows.append({"text": text, "label": CLASS_NAMES[self.train_labels[i]]})
        return pd.DataFrame(rows).to_html(index=False, classes="preview-table")

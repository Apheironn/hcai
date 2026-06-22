from collections import defaultdict

import numpy as np

from .dataset import CLASS_NAMES


class ExpertProfile:
    def __init__(self):
        self.correct = defaultdict(int)
        self.total = defaultdict(int)

    def update(self, true_labels, expert_preds):
        for y, pred in zip(true_labels, expert_preds):
            self.total[int(y)] += 1
            if int(y) == int(pred):
                self.correct[int(y)] += 1

    def per_class_accuracy(self):
        result = {}
        for idx, name in enumerate(CLASS_NAMES):
            if self.total[idx] == 0:
                result[name] = None
            else:
                result[name] = round(self.correct[idx] / self.total[idx], 4)
        return result

    def summary(self):
        parts = []
        for name, acc in self.per_class_accuracy().items():
            if acc is None:
                parts.append(f"{name}: no queries")
            else:
                parts.append(f"{name}: {acc}")
        return "; ".join(parts)

    @property
    def n_queries(self):
        return sum(self.total.values())

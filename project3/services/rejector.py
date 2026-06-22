import os

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


class Rejector:
    def __init__(self, model):
        self.model = model

    @classmethod
    def train(cls, vectorizer, texts, defer_labels, save_path=None):
        labels = np.asarray(defer_labels, dtype=int)
        if len(labels) == 0 or len(np.unique(labels)) < 2:
            return cls.empty()
        x = vectorizer.transform(texts)
        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(x, labels)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(model, save_path)
        return cls(model)

    @classmethod
    def load(cls, path):
        return cls(joblib.load(path))

    @classmethod
    def empty(cls):
        return cls(None)

    @property
    def is_trained(self):
        return self.model is not None

    def defer_proba(self, vectorizer, texts):
        if not self.is_trained:
            return np.zeros(len(texts))
        return self.model.predict_proba(vectorizer.transform(texts))[:, 1]

    def should_defer(self, vectorizer, texts):
        if not self.is_trained:
            return np.zeros(len(texts), dtype=bool)
        return self.model.predict(vectorizer.transform(texts)).astype(bool)

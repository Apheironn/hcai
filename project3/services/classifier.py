import os

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class BaselineClassifier:
    def __init__(self, pipeline, test_accuracy):
        self.pipeline = pipeline
        self.test_accuracy = test_accuracy

    @classmethod
    def train(cls, dataset, save_path=None):
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        pipeline.fit(dataset.train_texts, dataset.train_labels)
        preds = pipeline.predict(dataset.test_texts)
        acc = float(accuracy_score(dataset.test_labels, preds))

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(pipeline, save_path)

        return cls(pipeline, round(acc, 4))

    @classmethod
    def load(cls, path, test_accuracy):
        return cls(joblib.load(path), test_accuracy)

    @property
    def vectorizer(self):
        return self.pipeline.named_steps["tfidf"]

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_proba(self, texts):
        return self.pipeline.predict_proba(texts)

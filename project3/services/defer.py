from dataclasses import asdict, dataclass

import numpy as np

from .rejector import Rejector


@dataclass
class DeferMetrics:
    team_accuracy: float
    classifier_accuracy: float
    expert_accuracy: float
    defer_rate: float
    defer_precision: float
    defer_recall: float

    def to_dict(self):
        return asdict(self)


def build_defer_labels(classifier_preds, expert_preds, true_labels):
    labels = np.zeros(len(true_labels), dtype=int)
    for i, (cp, ep, y) in enumerate(zip(classifier_preds, expert_preds, true_labels)):
        if cp == y:
            labels[i] = 0
        elif ep == y:
            labels[i] = 1
        else:
            labels[i] = 0
    return labels


class DeferSystem:
    def __init__(self, classifier, rejector, expert):
        self.classifier = classifier
        self.rejector = rejector
        self.expert = expert

    def predict(self, texts):
        clf_preds = self.classifier.predict(texts)
        expert_preds = self.expert.predict_batch(texts)
        defer_mask = self.rejector.should_defer(self.classifier.vectorizer, texts)
        team = clf_preds.copy()
        team[defer_mask] = expert_preds[defer_mask]
        return team, defer_mask

    def evaluate(self, dataset) -> DeferMetrics:
        texts = dataset.test_texts
        labels = dataset.test_labels
        clf_preds = self.classifier.predict(texts)
        expert_preds = self.expert.predict_batch(texts)
        team_preds, defer_mask = self.predict(texts)

        team_acc = float(np.mean(team_preds == labels))
        clf_acc = float(np.mean(clf_preds == labels))
        exp_acc = float(np.mean(expert_preds == labels))
        defer_rate = float(np.mean(defer_mask))

        help_mask = (expert_preds == labels) & (clf_preds != labels)
        if defer_mask.sum() == 0:
            defer_prec = 0.0
        else:
            defer_prec = float(np.mean(help_mask[defer_mask]))
        if help_mask.sum() == 0:
            defer_rec = 0.0
        else:
            defer_rec = float(np.mean(defer_mask[help_mask]))

        return DeferMetrics(
            team_accuracy=round(team_acc, 4),
            classifier_accuracy=round(clf_acc, 4),
            expert_accuracy=round(exp_acc, 4),
            defer_rate=round(defer_rate, 4),
            defer_precision=round(defer_prec, 4),
            defer_recall=round(defer_rec, 4),
        )

    @classmethod
    def train_full(cls, dataset, classifier, expert, save_path=None):
        clf_train = classifier.predict(dataset.train_texts)
        exp_train = expert.predict_batch(dataset.train_texts)
        defer_labels = build_defer_labels(clf_train, exp_train, dataset.train_labels)
        rejector = Rejector.train(
            classifier.vectorizer,
            dataset.train_texts,
            defer_labels,
            save_path=save_path,
        )
        return cls(classifier, rejector, expert)

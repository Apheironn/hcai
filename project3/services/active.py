from dataclasses import dataclass

import numpy as np

from .defer import DeferSystem, build_defer_labels
from .expert_profile import ExpertProfile
from .rejector import Rejector


@dataclass
class ActiveStep:
    n_queries: int
    team_accuracy: float
    defer_rate: float
    expert_profile_summary: str


class ActiveLearner:
    @classmethod
    def run(cls, dataset, classifier, expert, budget, batch_size, strategy="entropy"):
        n_train = dataset.n_train
        queried = np.zeros(n_train, dtype=bool)
        query_texts = []
        defer_labels = []
        profile = ExpertProfile()
        steps = []

        clf_only = DeferSystem(classifier, Rejector.empty(), expert)
        steps.append(ActiveStep(0, clf_only.evaluate(dataset).team_accuracy, 0.0, profile.summary()))

        remaining = budget
        half = budget // 2

        while remaining > 0:
            batch = min(batch_size, remaining)
            pool = np.where(~queried)[0]
            if len(pool) == 0:
                break

            n_done = budget - remaining
            if strategy == "random":
                chosen = np.random.default_rng(42 + n_done).choice(pool, size=min(batch, len(pool)), replace=False)
            elif n_done < half:
                proba = classifier.predict_proba([dataset.train_texts[i] for i in pool])
                entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
                order = np.argsort(-entropy)
                chosen = pool[order[: min(batch, len(pool))]]
            else:
                rejector = Rejector.train(classifier.vectorizer, query_texts, defer_labels) if query_texts else Rejector.empty()
                if rejector.is_trained:
                    proba = rejector.defer_proba(classifier.vectorizer, [dataset.train_texts[i] for i in pool])
                    uncertainty = -np.abs(proba - 0.5)
                    order = np.argsort(uncertainty)
                    chosen = pool[order[: min(batch, len(pool))]]
                else:
                    proba = classifier.predict_proba([dataset.train_texts[i] for i in pool])
                    entropy = -np.sum(proba * np.log(proba + 1e-12), axis=1)
                    order = np.argsort(-entropy)
                    chosen = pool[order[: min(batch, len(pool))]]

            batch_texts = [dataset.train_texts[i] for i in chosen]
            batch_y = dataset.train_labels[chosen]
            batch_expert = expert.predict_batch(batch_texts)
            batch_clf = classifier.predict(batch_texts)
            batch_defer = build_defer_labels(batch_clf, batch_expert, batch_y)

            query_texts.extend(batch_texts)
            defer_labels.extend(batch_defer.tolist())
            profile.update(batch_y, batch_expert)
            queried[chosen] = True
            remaining -= len(chosen)

            rejector = Rejector.train(classifier.vectorizer, query_texts, defer_labels)
            system = DeferSystem(classifier, rejector, expert)
            metrics = system.evaluate(dataset)
            steps.append(
                ActiveStep(
                    profile.n_queries,
                    metrics.team_accuracy,
                    metrics.defer_rate,
                    profile.summary(),
                )
            )

        return steps

import os
from dataclasses import asdict, dataclass

import joblib
import numpy as np
from django.conf import settings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


TREE_GRID = [2, 3, 4, 6, 8, 12, 20, None]
LOGREG_GRID = [0.001, 0.01, 0.1, 1, 10, 100]


@dataclass
class ModelCandidate:
    id: int
    model_type: str
    param_label: str
    param_value: str
    test_accuracy: float
    complexity: float
    joblib_path: str

    def to_dict(self):
        return asdict(self)


class ModelBank:
    def __init__(self, candidates):
        self.candidates = candidates

    @classmethod
    def train(cls, dataset, cache_key):
        cache_dir = os.path.join(settings.MEDIA_ROOT, "model_cache", cache_key)
        os.makedirs(cache_dir, exist_ok=True)

        candidates = []
        cid = 0

        for value in TREE_GRID:
            kwargs = {"random_state": 42}
            label = "max_leaf_nodes"
            if value is not None:
                kwargs["max_leaf_nodes"] = value
                param_value = str(value)
            else:
                param_value = "None"

            model = DecisionTreeClassifier(**kwargs)
            model.fit(dataset.x_train, dataset.y_train)
            acc = float(accuracy_score(dataset.y_test, model.predict(dataset.x_test)))
            complexity = float(model.get_n_leaves())
            path = os.path.join(cache_dir, f"tree_{cid}.joblib")
            joblib.dump(model, path)
            candidates.append(
                ModelCandidate(
                    id=cid,
                    model_type="tree",
                    param_label=label,
                    param_value=param_value,
                    test_accuracy=round(acc, 4),
                    complexity=complexity,
                    joblib_path=path,
                )
            )
            cid += 1

        for value in LOGREG_GRID:
            model = LogisticRegression(C=value, max_iter=5000, random_state=42, solver="lbfgs")
            model.fit(dataset.x_train, dataset.y_train)
            acc = float(accuracy_score(dataset.y_test, model.predict(dataset.x_test)))
            complexity = float(np.sum(np.abs(model.coef_)))
            path = os.path.join(cache_dir, f"logreg_{cid}.joblib")
            joblib.dump(model, path)
            candidates.append(
                ModelCandidate(
                    id=cid,
                    model_type="logreg",
                    param_label="C",
                    param_value=str(value),
                    test_accuracy=round(acc, 4),
                    complexity=round(complexity, 4),
                    joblib_path=path,
                )
            )
            cid += 1

        return cls(candidates)

    def for_type(self, model_type):
        return [c for c in self.candidates if c.model_type == model_type]

    def load_model(self, candidate):
        return joblib.load(candidate.joblib_path)

    def to_session(self):
        return [c.to_dict() for c in self.candidates]

    @classmethod
    def from_session(cls, session_data):
        candidates = [ModelCandidate(**item) for item in session_data]
        return cls(candidates)

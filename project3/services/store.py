import os

from django.conf import settings

from .classifier import BaselineClassifier
from .defer import DeferSystem
from .expert import TopicKeywordExpert

SESSION_KEY = "project3_store"

# The baseline classifier is deterministic and expensive to fit (TF-IDF + LR on
# ~120k articles). Train it once per process and share it across every session,
# reloading from disk if a previous process already wrote it. This keeps
# "Reset session" and new visitors fast instead of retraining each time.
_SHARED_DIR = os.path.join(settings.MEDIA_ROOT, "model_cache", "project3", "shared")
_SHARED_CLF_PATH = os.path.join(_SHARED_DIR, "classifier.joblib")
_SHARED_CLASSIFIER = None


def _shared_classifier(dataset):
    global _SHARED_CLASSIFIER
    if _SHARED_CLASSIFIER is None:
        if os.path.isfile(_SHARED_CLF_PATH):
            import joblib
            from sklearn.metrics import accuracy_score

            pipeline = joblib.load(_SHARED_CLF_PATH)
            acc = round(
                float(accuracy_score(dataset.test_labels, pipeline.predict(dataset.test_texts))), 4
            )
            _SHARED_CLASSIFIER = BaselineClassifier(pipeline, acc)
        else:
            _SHARED_CLASSIFIER = BaselineClassifier.train(dataset, save_path=_SHARED_CLF_PATH)
    return _SHARED_CLASSIFIER


class ModelStore:
    def __init__(self, data, classifier, expert=None):
        self.data = data
        self.classifier = classifier
        self.expert = expert or TopicKeywordExpert()

    @classmethod
    def ensure(cls, request, dataset):
        if SESSION_KEY in request.session:
            return cls.from_session(request.session[SESSION_KEY])

        classifier = _shared_classifier(dataset)

        data = {
            "classifier_path": _SHARED_CLF_PATH,
            "baseline_accuracy": classifier.test_accuracy,
            "rejector_path": None,
            "defer_metrics": None,
            "defer_chart_url": None,
            "expert_report": None,
            "expert_chart_url": None,
            "active_chart_url": None,
            "active_final_accuracy": None,
            "active_random_final_accuracy": None,
            "human_labels": {},
            "human_batch_indices": [],
            "human_report": None,
            "human_chart_url": None,
            "defer_inspector_url": None,
        }
        request.session[SESSION_KEY] = data
        return cls(data, classifier)

    @classmethod
    def from_session(cls, data):
        global _SHARED_CLASSIFIER
        if _SHARED_CLASSIFIER is None:
            _SHARED_CLASSIFIER = BaselineClassifier.load(
                data["classifier_path"], data["baseline_accuracy"]
            )
        return cls(data, _SHARED_CLASSIFIER)

    def save(self, request):
        request.session[SESSION_KEY] = self.data
        request.session.modified = True

    def train_full_defer(self, dataset):
        rejector_path = os.path.join(os.path.dirname(self.data["classifier_path"]), "rejector.joblib")
        system = DeferSystem.train_full(dataset, self.classifier, self.expert, save_path=rejector_path)
        self.data["rejector_path"] = rejector_path
        self.data["defer_metrics"] = system.evaluate(dataset).to_dict()
        return system

    def load_defer_system(self, dataset):
        if not self.data.get("rejector_path"):
            return DeferSystem.train_full(dataset, self.classifier, self.expert)
        from .rejector import Rejector

        rejector = Rejector.load(self.data["rejector_path"])
        return DeferSystem(self.classifier, rejector, self.expert)

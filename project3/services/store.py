import os

from django.conf import settings

from .classifier import BaselineClassifier
from .defer import DeferSystem
from .expert import TopicKeywordExpert

SESSION_KEY = "project3_store"


class ModelStore:
    def __init__(self, data, classifier, expert=None):
        self.data = data
        self.classifier = classifier
        self.expert = expert or TopicKeywordExpert()

    @classmethod
    def ensure(cls, request, dataset):
        if SESSION_KEY in request.session:
            return cls.from_session(request.session[SESSION_KEY])

        if not request.session.session_key:
            request.session.save()
        cache_dir = os.path.join(
            settings.MEDIA_ROOT, "model_cache", "project3", request.session.session_key
        )
        clf_path = os.path.join(cache_dir, "classifier.joblib")
        classifier = BaselineClassifier.train(dataset, save_path=clf_path)

        data = {
            "classifier_path": clf_path,
            "baseline_accuracy": classifier.test_accuracy,
            "rejector_path": None,
            "defer_metrics": None,
            "defer_chart_url": None,
            "expert_report": None,
            "expert_chart_url": None,
            "active_chart_url": None,
            "active_final_accuracy": None,
            "active_random_final_accuracy": None,
        }
        request.session[SESSION_KEY] = data
        return cls(data, classifier)

    @classmethod
    def from_session(cls, data):
        classifier = BaselineClassifier.load(data["classifier_path"], data["baseline_accuracy"])
        return cls(data, classifier)

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

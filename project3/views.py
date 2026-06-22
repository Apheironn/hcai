import os
import uuid

from django.conf import settings
from django.http import FileResponse
from django.shortcuts import render
from django.urls import reverse

from .forms import ActiveLearningForm
from .services.active import ActiveLearner
from .services.dataset import AgNewsDataset, CLASS_NAMES
from .services.plots import ResultPlots
from .services.report import ReportBuilder
from .services.store import ModelStore, SESSION_KEY


def _build_context(request, dataset, store, error=None, training=False):
    data = store.data
    return {
        "class_names": CLASS_NAMES,
        "n_train": dataset.n_train,
        "n_test": dataset.n_test,
        "preview_html": dataset.preview(),
        "class_chart_url": ResultPlots.class_distribution(dataset.class_counts()),
        "baseline_accuracy": data.get("baseline_accuracy"),
        "training": training,
        "expert_report": data.get("expert_report"),
        "expert_chart_url": data.get("expert_chart_url"),
        "defer_metrics": data.get("defer_metrics"),
        "defer_chart_url": data.get("defer_chart_url"),
        "active_chart_url": data.get("active_chart_url"),
        "active_final_accuracy": data.get("active_final_accuracy"),
        "active_random_final_accuracy": data.get("active_random_final_accuracy"),
        "oracle_team_accuracy": data.get("defer_metrics", {}).get("team_accuracy") if data.get("defer_metrics") else None,
        "active_form": ActiveLearningForm(),
        "report_url": reverse("project3:report"),
        "error": error,
    }


def index(request):
    error = None
    try:
        dataset = AgNewsDataset.load()
        store = ModelStore.ensure(request, dataset)

        if request.method == "POST":
            action = request.POST.get("action")

            if action == "run_expert":
                report = store.expert.analyze(dataset)
                clf_preds = store.classifier.predict(dataset.test_texts)
                clf_per_class = ResultPlots.per_class_accuracy(clf_preds, dataset.test_labels)
                store.data["expert_report"] = {
                    "name": report.name,
                    "overall_accuracy": report.overall_accuracy,
                    "per_class_accuracy": report.per_class_accuracy,
                    "classifier_per_class_accuracy": clf_per_class,
                    "strengths": report.strengths,
                    "weaknesses": report.weaknesses,
                }
                store.data["expert_chart_url"] = ResultPlots.expert_vs_classifier(
                    report.per_class_accuracy, clf_per_class
                )
                store.save(request)

            elif action == "run_defer":
                store.train_full_defer(dataset)
                store.data["defer_chart_url"] = ResultPlots.defer_summary(store.data["defer_metrics"])
                store.save(request)

            elif action == "run_active":
                form = ActiveLearningForm(request.POST)
                if form.is_valid():
                    budget = form.cleaned_data["budget"]
                    batch_size = form.cleaned_data["batch_size"]
                    entropy_steps = ActiveLearner.run(
                        dataset, store.classifier, store.expert, budget, batch_size, strategy="entropy"
                    )
                    random_steps = ActiveLearner.run(
                        dataset, store.classifier, store.expert, budget, batch_size, strategy="random"
                    )
                    oracle = store.data.get("defer_metrics", {}).get("team_accuracy")
                    store.data["active_chart_url"] = ResultPlots.active_curves(
                        entropy_steps,
                        random_steps,
                        baseline=store.data.get("baseline_accuracy"),
                        oracle=oracle,
                    )
                    store.data["active_final_accuracy"] = entropy_steps[-1].team_accuracy
                    store.data["active_random_final_accuracy"] = random_steps[-1].team_accuracy
                    store.data["active_budget"] = budget
                    store.data["active_batch_size"] = batch_size
                    store.save(request)
                else:
                    error = "Invalid active learning parameters."

        context = _build_context(request, dataset, store, error=error)
    except Exception as exc:
        context = {
            "class_names": CLASS_NAMES,
            "error": str(exc),
            "active_form": ActiveLearningForm(),
            "report_url": reverse("project3:report"),
        }

    return render(request, "project3/index.html", context)


def download_report(request):
    if SESSION_KEY not in request.session:
        return render(
            request,
            "project3/index.html",
            {
                "error": "Open Project 3 and run experiments before downloading the report.",
                "active_form": ActiveLearningForm(),
                "report_url": reverse("project3:report"),
            },
            status=400,
        )

    dataset = AgNewsDataset.load()
    store = ModelStore.from_session(request.session[SESSION_KEY])
    report_dir = os.path.join(settings.MEDIA_ROOT, "reports")
    output_path = os.path.join(report_dir, f"project3_{uuid.uuid4().hex}.pdf")
    ReportBuilder.build(dataset, store.data, output_path)

    pdf_file = open(output_path, "rb")
    response = FileResponse(pdf_file, content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="project3_report.pdf"'
    return response

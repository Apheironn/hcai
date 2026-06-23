import os
import uuid

from django.conf import settings
from django.http import FileResponse
from django.shortcuts import redirect, render
from django.urls import reverse

from .forms import ActiveLearningForm
from .services.active import ActiveLearner
from .services.dataset import AgNewsDataset, CLASS_NAMES
from .services.expert import highlight_keywords
from .services.human_expert import HumanExpertSession
from .services.plots import ResultPlots
from .services.report import ReportBuilder
from .services.store import ModelStore, SESSION_KEY


def _workflow_status(data):
    return {
        "baseline": data.get("baseline_accuracy") is not None,
        "expert": bool(data.get("expert_report")),
        "defer": bool(data.get("defer_metrics")),
        "active": data.get("active_final_accuracy") is not None,
        "human": bool(data.get("human_report")),
    }


def _build_context(request, dataset, store, error=None):
    data = store.data
    batch_indices = data.get("human_batch_indices", [])
    human_done, human_total = HumanExpertSession.batch_progress(store)
    batch_articles = []
    for idx in batch_indices:
        text = dataset.train_texts[idx]
        batch_articles.append(
            {
                "index": idx,
                "text_html": highlight_keywords(text),
                "labeled": str(idx) in data.get("human_labels", {}),
                "label": CLASS_NAMES[data["human_labels"][str(idx)]] if str(idx) in data.get("human_labels", {}) else None,
            }
        )

    return {
        "class_names": CLASS_NAMES,
        "n_train": dataset.n_train,
        "n_test": dataset.n_test,
        "preview_html": dataset.preview(),
        "class_chart_url": ResultPlots.class_distribution(dataset.class_counts()),
        "baseline_accuracy": data.get("baseline_accuracy"),
        "expert_report": data.get("expert_report"),
        "expert_chart_url": data.get("expert_chart_url"),
        "defer_metrics": data.get("defer_metrics"),
        "defer_chart_url": data.get("defer_chart_url"),
        "defer_inspector_url": data.get("defer_inspector_url"),
        "defer_inspector_rows": data.get("defer_inspector_rows"),
        "active_chart_url": data.get("active_chart_url"),
        "active_final_accuracy": data.get("active_final_accuracy"),
        "active_random_final_accuracy": data.get("active_random_final_accuracy"),
        "oracle_team_accuracy": data.get("defer_metrics", {}).get("team_accuracy") if data.get("defer_metrics") else None,
        "human_report": data.get("human_report"),
        "human_chart_url": data.get("human_chart_url"),
        "batch_articles": batch_articles,
        "human_done": human_done,
        "human_total": human_total,
        "workflow": _workflow_status(data),
        "active_form": ActiveLearningForm(),
        "report_url": reverse("project3:report"),
        "report_ready": data.get("baseline_accuracy") is not None,
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
                system = store.load_defer_system(dataset)
                chart_url, rows = ResultPlots.defer_inspector(dataset, system)
                store.data["defer_inspector_url"] = chart_url
                store.data["defer_inspector_rows"] = rows
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

            elif action == "load_human_batch":
                HumanExpertSession.start_batch(dataset, store)
                store.save(request)

            elif action == "label_expert":
                try:
                    idx = int(request.POST.get("article_index"))
                    class_name = request.POST.get("class_name")
                    HumanExpertSession.record_label(store, idx, class_name)
                    store.save(request)
                except (TypeError, ValueError) as exc:
                    error = str(exc)

            elif action == "finish_human":
                try:
                    report = HumanExpertSession.build_report(dataset, store, store.expert)
                    store.data["human_chart_url"] = ResultPlots.human_vs_simulated(report)
                    store.save(request)
                except ValueError as exc:
                    error = str(exc)

            elif action == "reset_session":
                request.session.pop(SESSION_KEY, None)
                return redirect("project3:index")

        context = _build_context(request, dataset, store, error=error)
    except Exception as exc:
        context = {
            "class_names": CLASS_NAMES,
            "error": str(exc),
            "active_form": ActiveLearningForm(),
            "report_url": reverse("project3:report"),
            "report_ready": False,
            "workflow": {},
            "batch_articles": [],
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
                "report_ready": False,
                "workflow": {},
                "batch_articles": [],
            },
            status=400,
        )

    dataset = AgNewsDataset.load()
    store = ModelStore.from_session(request.session[SESSION_KEY])
    report_dir = os.path.join(settings.MEDIA_ROOT, "reports")
    output_path = os.path.join(report_dir, f"project3_{uuid.uuid4().hex}.pdf")
    ReportBuilder.build(dataset, store.data, output_path)

    return FileResponse(open(output_path, "rb"), content_type="application/pdf", as_attachment=True, filename="project3_report.pdf")

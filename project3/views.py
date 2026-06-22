from django.shortcuts import render
from django.templatetags.static import static

from .forms import ActiveLearningForm
from .services.active import ActiveLearner
from .services.dataset import AgNewsDataset, CLASS_NAMES
from .services.plots import ResultPlots
from .services.store import ModelStore


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
        "active_chart_url": data.get("active_chart_url"),
        "active_final_accuracy": data.get("active_final_accuracy"),
        "active_random_final_accuracy": data.get("active_random_final_accuracy"),
        "oracle_team_accuracy": data.get("defer_metrics", {}).get("team_accuracy") if data.get("defer_metrics") else None,
        "active_form": ActiveLearningForm(),
        "report_url": static("project3/report.pdf"),
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
                store.data["expert_report"] = {
                    "name": report.name,
                    "overall_accuracy": report.overall_accuracy,
                    "per_class_accuracy": report.per_class_accuracy,
                    "strengths": report.strengths,
                    "weaknesses": report.weaknesses,
                }
                store.data["expert_chart_url"] = ResultPlots.expert_per_class(report.per_class_accuracy)
                store.save(request)

            elif action == "run_defer":
                store.train_full_defer(dataset)
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
                    store.data["active_chart_url"] = ResultPlots.active_curves(entropy_steps, random_steps)
                    store.data["active_final_accuracy"] = entropy_steps[-1].team_accuracy
                    store.data["active_random_final_accuracy"] = random_steps[-1].team_accuracy
                    store.save(request)
                else:
                    error = "Invalid active learning parameters."

        context = _build_context(request, dataset, store, error=error)
    except Exception as exc:
        context = {
            "class_names": CLASS_NAMES,
            "error": str(exc),
            "active_form": ActiveLearningForm(),
            "report_url": static("project3/report.pdf"),
        }

    return render(request, "project3/index.html", context)

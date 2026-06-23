import json

from django.http import JsonResponse
from django.shortcuts import render

from .forms import CounterfactualForm, EffectForm
from .services.coef_plot import CoefficientPlotter
from .services.counterfactual import CounterfactualFinder
from .services.dataset import NUMERIC_FEATURES, PenguinDataset
from .services.effects import EffectPlots
from .services.model_bank import ModelBank
from .services.pareto_plot import ParetoPlotter
from .services.plot_style import PlotStyle
from .services.selector import ModelSelector
from .services.tree_plot import TreePlotter

SESSION_BANK = "project2_bank"
DEFAULT_MODEL_TYPE = "tree"
DEFAULT_LAMBDA = 0.0


def _get_dataset():
    return PenguinDataset.load()


def _ensure_bank(request, dataset):
    if SESSION_BANK not in request.session:
        if not request.session.session_key:
            request.session.save()
        cache_key = request.session.session_key
        bank = ModelBank.train(dataset, cache_key)
        request.session[SESSION_BANK] = bank.to_session()
    return ModelBank.from_session(request.session[SESSION_BANK])


def _parse_controls(request):
    if request.method == "POST":
        model_type = request.POST.get("model_type", DEFAULT_MODEL_TYPE)
        try:
            lambda_value = float(request.POST.get("lambda_value", DEFAULT_LAMBDA))
        except (TypeError, ValueError):
            lambda_value = DEFAULT_LAMBDA
    else:
        model_type = request.GET.get("model_type", DEFAULT_MODEL_TYPE)
        try:
            lambda_value = float(request.GET.get("lambda", DEFAULT_LAMBDA))
        except (TypeError, ValueError):
            lambda_value = DEFAULT_LAMBDA

    if model_type not in ("tree", "logreg"):
        model_type = DEFAULT_MODEL_TYPE
    lambda_value = max(0.0, min(lambda_value, 0.05))
    return model_type, lambda_value


def _render_plot(bank, dataset, candidate):
    model = bank.load_model(candidate)
    if candidate.model_type == "tree":
        return TreePlotter.render(model, dataset, candidate)
    return CoefficientPlotter.render(model, dataset, candidate)


def _build_context(request, dataset, bank, model_type, lambda_value, extra=None):
    candidates = bank.for_type(model_type)
    rows, selected = ModelSelector.scored_candidates(candidates, lambda_value)
    plot_url = _render_plot(bank, dataset, selected)
    score = ModelSelector.score(selected, lambda_value)
    pareto_url = ParetoPlotter.render(candidates, lambda_value, selected.id)

    context = {
        "dataset": dataset,
        "preview_html": dataset.preview(),
        "class_chart_url": dataset.class_distribution_plot(PlotStyle),
        "model_type": model_type,
        "lambda_value": lambda_value,
        "selected": selected,
        "candidate_rows": rows,
        "plot_url": plot_url,
        "pareto_url": pareto_url,
        "score": round(score, 4),
        "param_text": f"{selected.param_label}={selected.param_value}",
        "row_choices": [(i, dataset.row_label(i)) for i in range(dataset.n_rows)],
        "class_names": dataset.class_names,
        "numeric_features": NUMERIC_FEATURES,
        "candidates_json": json.dumps(rows),
        "error": None,
        "counterfactual_results": None,
        "cf_scatter_url": None,
        "cf_instance": None,
        "effect_plot_url": None,
        "effect_note": None,
    }
    if extra:
        context.update(extra)
    return context


def index(request):
    dataset = _get_dataset()
    bank = _ensure_bank(request, dataset)
    model_type, lambda_value = _parse_controls(request)
    error = None
    counterfactual_results = None
    cf_scatter_url = None
    cf_instance = None
    effect_plot_url = None
    effect_note = None

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "counterfactual":
            form = CounterfactualForm(request.POST)
            if form.is_valid():
                try:
                    model_type = form.cleaned_data["model_type"]
                    lambda_value = form.cleaned_data["lambda_value"]
                    candidates = bank.for_type(model_type)
                    selected = ModelSelector.select(candidates, lambda_value)
                    model = bank.load_model(selected)
                    row_index = form.cleaned_data["row_index"]
                    original_raw, _ = dataset.row_raw(row_index)
                    found = CounterfactualFinder.find(
                        model,
                        dataset,
                        row_index,
                        form.cleaned_data["target_class"],
                    )
                    counterfactual_results = []
                    for item in found:
                        counterfactual_results.append(
                            {
                                "distance": item["distance"],
                                "predicted": item["predicted"],
                                "rows": CounterfactualFinder.build_diff_rows(original_raw, item),
                            }
                        )
                    if not counterfactual_results:
                        error = "No counterfactuals found. Try another example or model."
                    else:
                        cf_scatter_url = CounterfactualFinder.scatter_plot(
                            dataset, row_index, found
                        )
                        cf_instance = CounterfactualFinder.instance_summary(
                            dataset, row_index, model
                        )
                except ValueError as exc:
                    error = str(exc)
            else:
                error = "Invalid counterfactual request."

        elif action == "effects":
            form = EffectForm(request.POST)
            if form.is_valid():
                try:
                    model_type = form.cleaned_data["model_type"]
                    lambda_value = form.cleaned_data["lambda_value"]
                    candidates = bank.for_type(model_type)
                    selected = ModelSelector.select(candidates, lambda_value)
                    model = bank.load_model(selected)
                    effect_plot_url, effect_note = EffectPlots.render(
                        model,
                        dataset,
                        selected,
                        form.cleaned_data["feature"],
                        model_type,
                        lambda_value,
                    )
                except ValueError as exc:
                    error = str(exc)
            else:
                error = "Invalid effect plot request."

    context = _build_context(
        request,
        dataset,
        bank,
        model_type,
        lambda_value,
        {
            "error": error,
            "counterfactual_results": counterfactual_results,
            "cf_scatter_url": cf_scatter_url,
            "cf_instance": cf_instance,
            "effect_plot_url": effect_plot_url,
            "effect_note": effect_note,
        },
    )
    return render(request, "project2/index.html", context)


def preview_instance(request):
    dataset = _get_dataset()
    bank = _ensure_bank(request, dataset)
    model_type, lambda_value = _parse_controls(request)
    try:
        row_index = int(request.GET.get("row_index", 0))
    except (TypeError, ValueError):
        return JsonResponse({"error": "Invalid row index."}, status=400)

    try:
        candidates = bank.for_type(model_type)
        selected = ModelSelector.select(candidates, lambda_value)
        model = bank.load_model(selected)
        summary = CounterfactualFinder.instance_summary(dataset, row_index, model)
        raw = summary["raw"]
        return JsonResponse(
            {
                "label": summary["label"],
                "predicted": summary["predicted"],
                "island": raw.get("island"),
                "sex": raw.get("sex"),
                "bill_length_mm": raw.get("bill_length_mm"),
                "bill_depth_mm": raw.get("bill_depth_mm"),
            }
        )
    except (ValueError, IndexError) as exc:
        return JsonResponse({"error": str(exc)}, status=400)


def select_model(request):
    dataset = _get_dataset()
    bank = _ensure_bank(request, dataset)
    model_type, lambda_value = _parse_controls(request)

    try:
        candidates = bank.for_type(model_type)
        rows, selected = ModelSelector.scored_candidates(candidates, lambda_value)
        plot_url = _render_plot(bank, dataset, selected)
        pareto_url = ParetoPlotter.render(candidates, lambda_value, selected.id)
        return JsonResponse(
            {
                "accuracy": selected.test_accuracy,
                "complexity": selected.complexity,
                "score": round(ModelSelector.score(selected, lambda_value), 4),
                "param": f"{selected.param_label}={selected.param_value}",
                "plot_url": plot_url,
                "pareto_url": pareto_url,
                "selected_id": selected.id,
                "candidates": rows,
            }
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

import json

from django.shortcuts import render

from .forms import PlotForm, TrainForm, UploadForm
from .services.dataset import Dataset
from .services.plot import PlotBuilder
from .services.trainer import ModelTrainer

SESSION_KEY = "dataset"


def _load_dataset(request, problem_type="auto"):
    if SESSION_KEY not in request.session:
        raise ValueError("Upload a CSV first.")
    return Dataset.from_session(request.session[SESSION_KEY], problem_type)


def index(request):
    upload_form = UploadForm()
    plot_form = None
    train_form = None
    error = None
    dataset = None
    preview_html = None
    plot_url = None
    hist_url = None
    class_chart_url = None
    train_result = None
    train_plot_url = None
    model_choices = []
    metric_choices = []
    param_name = ""
    model_config_json = "{}"
    selected_model = ""
    selected_metric = ""
    selected_params = ""

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "upload":
            upload_form = UploadForm(request.POST, request.FILES)
            if upload_form.is_valid():
                try:
                    dataset, path = Dataset.from_upload(upload_form.cleaned_data["file"])
                    request.session[SESSION_KEY] = dataset.to_session(path)
                    preview_html = dataset.preview()
                    class_chart_url = PlotBuilder.class_distribution(dataset)
                except ValueError as exc:
                    error = str(exc)
            else:
                error = "Select a valid CSV file."

        elif action == "load_sample":
            try:
                dataset, path = Dataset.from_sample()
                request.session[SESSION_KEY] = dataset.to_session(path)
                preview_html = dataset.preview()
                class_chart_url = PlotBuilder.class_distribution(dataset)
            except ValueError as exc:
                error = str(exc)

        elif action == "plot":
            plot_form = PlotForm(request.POST)
            if plot_form.is_valid():
                try:
                    problem_type = plot_form.cleaned_data["problem_type"]
                    dataset = _load_dataset(request, problem_type)
                    request.session[SESSION_KEY]["problem_type"] = dataset.problem_type
                    plot_url = PlotBuilder.build(
                        dataset,
                        plot_form.cleaned_data["x_feature"],
                        plot_form.cleaned_data["y_feature"] or None,
                    )
                    preview_html = dataset.preview()
                    class_chart_url = PlotBuilder.class_distribution(dataset)
                except ValueError as exc:
                    error = str(exc)

        elif action == "histogram":
            plot_form = PlotForm(request.POST)
            if plot_form.is_valid():
                try:
                    dataset = _load_dataset(request, plot_form.cleaned_data["problem_type"])
                    hist_url = PlotBuilder.histogram(dataset, plot_form.cleaned_data["x_feature"])
                    preview_html = dataset.preview()
                    class_chart_url = PlotBuilder.class_distribution(dataset)
                except ValueError as exc:
                    error = str(exc)

        elif action == "train":
            train_form = TrainForm(request.POST)
            if train_form.is_valid():
                try:
                    dataset = _load_dataset(request)
                    train_result = ModelTrainer.run(
                        dataset,
                        train_form.cleaned_data["model"],
                        train_form.cleaned_data["test_size"],
                        train_form.cleaned_data["param_values"],
                        train_form.cleaned_data["metric"],
                    )
                    train_plot_url = PlotBuilder.param_sweep(
                        train_result["results"],
                        train_result["param_name"],
                        train_result["metric"],
                        train_result["lower_is_better"],
                    )
                    if dataset.problem_type == "classification":
                        train_plot_url_extra = PlotBuilder.confusion_matrix_plot(
                            train_result["y_test"],
                            train_result["y_pred"],
                            train_result["label_classes"],
                        )
                        train_result["diagnostic_plot_url"] = train_plot_url_extra
                    else:
                        train_result["diagnostic_plot_url"] = PlotBuilder.residuals(
                            train_result["y_test"], train_result["y_pred"]
                        )
                    preview_html = dataset.preview()
                    class_chart_url = PlotBuilder.class_distribution(dataset)
                except ValueError as exc:
                    error = str(exc)
                    try:
                        dataset = _load_dataset(request)
                        preview_html = dataset.preview()
                        class_chart_url = PlotBuilder.class_distribution(dataset)
                    except ValueError:
                        pass

    if dataset is None and SESSION_KEY in request.session:
        try:
            dataset = _load_dataset(request)
            preview_html = dataset.preview()
            class_chart_url = PlotBuilder.class_distribution(dataset)
        except ValueError as exc:
            error = str(exc)
            request.session.pop(SESSION_KEY, None)

    if dataset:
        if plot_form is None:
            plot_form = PlotForm(initial={"problem_type": dataset.problem_type})

        model_choices = list(ModelTrainer.models_for(dataset.problem_type).keys())
        metric_choices = ModelTrainer.metrics_for(dataset.problem_type)
        model_config_json = json.dumps(ModelTrainer.model_config(dataset.problem_type))

        if train_form is None:
            selected_model = model_choices[0]
            selected_metric = metric_choices[0]
            selected_params = ModelTrainer.default_values_for(
                selected_model, dataset.problem_type
            )
            param_name = ModelTrainer.param_name_for(selected_model, dataset.problem_type) or ""
            train_form = TrainForm(
                initial={
                    "model": selected_model,
                    "test_size": 80,
                    "metric": selected_metric,
                    "param_values": selected_params,
                }
            )
        else:
            selected_model = train_form.data.get("model", model_choices[0])
            selected_metric = train_form.data.get("metric", metric_choices[0])
            selected_params = train_form.data.get(
                "param_values",
                ModelTrainer.default_values_for(selected_model, dataset.problem_type),
            )
            param_name = ModelTrainer.param_name_for(selected_model, dataset.problem_type) or ""

    return render(
        request,
        "project1/index.html",
        {
            "upload_form": upload_form,
            "plot_form": plot_form,
            "train_form": train_form,
            "dataset": dataset,
            "preview_html": preview_html,
            "plot_url": plot_url,
            "hist_url": hist_url,
            "class_chart_url": class_chart_url,
            "train_result": train_result,
            "train_plot_url": train_plot_url,
            "model_choices": model_choices,
            "metric_choices": metric_choices,
            "param_name": param_name,
            "model_config_json": model_config_json,
            "selected_model": selected_model,
            "selected_metric": selected_metric,
            "selected_params": selected_params,
            "error": error,
        },
    )

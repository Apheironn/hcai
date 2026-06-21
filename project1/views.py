from django.shortcuts import render

from .forms import PlotForm, UploadForm
from .services.dataset import Dataset
from .services.plot import PlotBuilder

SESSION_KEY = "dataset"


def _load_dataset(request, problem_type="auto"):
    if SESSION_KEY not in request.session:
        raise ValueError("Upload a CSV first.")
    return Dataset.from_session(request.session[SESSION_KEY], problem_type)


def index(request):
    upload_form = UploadForm()
    plot_form = None
    error = None
    dataset = None
    preview_html = None
    plot_url = None

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "upload":
            upload_form = UploadForm(request.POST, request.FILES)
            if upload_form.is_valid():
                try:
                    dataset, path = Dataset.from_upload(upload_form.cleaned_data["file"])
                    request.session[SESSION_KEY] = dataset.to_session(path)
                    preview_html = dataset.preview()
                except ValueError as exc:
                    error = str(exc)
            else:
                error = "Select a valid CSV file."

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
                except ValueError as exc:
                    error = str(exc)

    if dataset is None and SESSION_KEY in request.session:
        try:
            dataset = _load_dataset(request)
            preview_html = dataset.preview()
        except ValueError as exc:
            error = str(exc)
            request.session.pop(SESSION_KEY, None)

    if dataset and plot_form is None:
        plot_form = PlotForm(initial={"problem_type": dataset.problem_type})

    return render(
        request,
        "project1/index.html",
        {
            "upload_form": upload_form,
            "plot_form": plot_form,
            "dataset": dataset,
            "preview_html": preview_html,
            "plot_url": plot_url,
            "error": error,
        },
    )

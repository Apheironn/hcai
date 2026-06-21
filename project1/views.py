from django.shortcuts import render

from .forms import UploadForm
from .services.dataset import Dataset

SESSION_KEY = "dataset"


def index(request):
    upload_form = UploadForm()
    error = None
    dataset = None
    preview_html = None

    if request.method == "POST" and request.POST.get("action") == "upload":
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
    elif SESSION_KEY in request.session:
        try:
            dataset = Dataset.from_session(request.session[SESSION_KEY])
            preview_html = dataset.preview()
        except ValueError as exc:
            error = str(exc)
            request.session.pop(SESSION_KEY, None)

    return render(
        request,
        "project1/index.html",
        {
            "upload_form": upload_form,
            "dataset": dataset,
            "preview_html": preview_html,
            "error": error,
        },
    )

import os

from django.conf import settings
from django.http import FileResponse, Http404
from django.shortcuts import redirect, render

from .forms import StartStudyForm, SurveyForm
from .services.dataset import MovieDataset
from .services.plots import StudyPlots
from .services.report import FILENAME, StudyReport
from .services.study import (
    HOLDOUT_PAIRS,
    PAIR_TASKS,
    RANK_SIZE,
    RANK_TASKS,
    SESSION_KEY,
    StudySession,
)


def _study_info():
    return {
        "pair_tasks": PAIR_TASKS,
        "rank_tasks": RANK_TASKS,
        "rank_size": RANK_SIZE,
        "holdout_pairs": HOLDOUT_PAIRS,
    }


def landing(request):
    error = None
    summary = None
    try:
        dataset = MovieDataset.load()
        summary = dataset.summary()

        if request.method == "POST":
            action = request.POST.get("action")
            if action == "start":
                form = StartStudyForm(request.POST)
                if form.is_valid():
                    participant = {
                        "code": form.cleaned_data["participant_code"].strip(),
                        "age_group": form.cleaned_data["age_group"],
                        "movie_frequency": form.cleaned_data["movie_frequency"],
                    }
                    StudySession.start(request, dataset, participant)
                    return redirect("project4:study")
                error = "Please fill in the participant code and confirm your consent."
            elif action == "reset":
                request.session.pop(SESSION_KEY, None)
                return redirect("project4:landing")
            else:
                form = StartStudyForm()
        else:
            form = StartStudyForm()
    except ValueError as exc:
        error = str(exc)
        form = StartStudyForm()

    session = StudySession.from_request(request)
    context = {
        "summary": summary,
        "form": form,
        "error": error,
        "has_session": session is not None,
        "session_finished": session.finished if session else False,
        **_study_info(),
    }
    return render(request, "project4/landing.html", context)


def study(request):
    session = StudySession.from_request(request)
    if session is None:
        return redirect("project4:landing")

    dataset = MovieDataset.load()
    error = None

    if request.method == "POST":
        try:
            error = _handle_submission(request, session, dataset)
        except ValueError as exc:
            error = str(exc)
        session.save(request)
        if error is None:
            return redirect("project4:results") if session.finished else redirect("project4:study")

    if session.finished:
        return redirect("project4:results")

    step = session.current()
    session.mark_started()
    session.save(request)

    done, total = session.progress()
    context = {
        "step": step,
        "cards": dataset.cards(step["items"]) if step.get("items") else [],
        "survey_form": SurveyForm() if step["kind"] == "survey" else None,
        "done": done,
        "total": total,
        "percent": int(100 * done / total) if total else 0,
        "participant": session.data["participant"],
        "error": error,
        **_study_info(),
    }
    return render(request, "project4/study.html", context)


def _handle_submission(request, session, dataset):
    action = request.POST.get("action")

    if action == "continue":
        session.advance()
    elif action == "choose":
        choice = int(request.POST.get("choice"))
        items = session.current()["items"]
        if choice not in items:
            raise ValueError("Unknown movie in the submitted choice.")
        other = [i for i in items if i != choice][0]
        session.record_choice([choice, other])
    elif action == "rank":
        raw = request.POST.get("order", "")
        ranking = [int(value) for value in raw.split(",") if value.strip()]
        session.record_choice(ranking)
    elif action == "skip":
        session.record_skip()
    elif action == "survey":
        form = SurveyForm(request.POST)
        if not form.is_valid():
            return "Please answer all three statements."
        session.record_survey(form.cleaned_data)
    else:
        raise ValueError("Unknown action.")
    return None


def results(request):
    session = StudySession.from_request(request)
    if session is None:
        return redirect("project4:landing")
    if not session.finished:
        return redirect("project4:study")

    dataset = MovieDataset.load()
    if not session.data.get("results"):
        results_data = session.build_results(dataset)
        session.data["weights_chart_url"] = StudyPlots.learned_weights(results_data)
        session.data["comparison_chart_url"] = StudyPlots.design_comparison(results_data)
        session.save(request)

    context = {
        "results": session.data["results"],
        "designs": [session.data["results"]["designs"][d] for d in session.data["results"]["order"]],
        "weights_chart_url": session.data.get("weights_chart_url"),
        "comparison_chart_url": session.data.get("comparison_chart_url"),
        "participant": session.data["participant"],
        **_study_info(),
    }
    return render(request, "project4/results.html", context)


def report(request):
    output_path = os.path.join(settings.MEDIA_ROOT, "reports", FILENAME)
    try:
        if not os.path.isfile(output_path):
            StudyReport.build(MovieDataset.load().summary(), output_path)
    except ValueError as exc:
        raise Http404(str(exc))
    return FileResponse(
        open(output_path, "rb"),
        content_type="application/pdf",
        as_attachment=True,
        filename="project4_study_design.pdf",
    )

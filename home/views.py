from django.shortcuts import render


def index(request):
    students = [
        {"name": "Mertcan Catak", "matriculation": "642815"},
    ]

    projects = [
        {
            "name": "Project 1",
            "url_name": "project1:index",
            "description": "Upload data, explore features, and train classifiers.",
            "chips": ["Grid search", "Confusion matrix", "Sample Iris"],
        },
        {
            "name": "Project 2",
            "url_name": "project2:index",
            "description": "Explainability, counterfactuals, and feature effects.",
            "chips": ["λ tradeoff", "Counterfactuals", "PDP & ALE"],
        },
        {
            "name": "Project 3",
            "url_name": "project3:index",
            "description": "Learning to defer and active expert querying.",
            "chips": ["Defer system", "Active learning", "Human expert"],
        },
    ]

    return render(
        request,
        "home/index.html",
        {"students": students, "projects": projects},
    )

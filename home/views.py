from django.shortcuts import render


def index(request):
    students = [
        {"name": "Mertcan Catak", "matriculation": "642815"},
    ]

    projects = [
        {
            "name": "Project 1",
            "title": "Supervised Learning",
            "url_name": "project1:index",
            "description": "Upload data, explore features, and train classifiers.",
            "blurb": (
                "Upload a CSV, visualise the features, then train and compare models "
                "with a train/test split and a hyperparameter sweep scored by the metric "
                "you choose."
            ),
            "chips": ["Grid search", "Confusion matrix", "Sample Iris"],
        },
        {
            "name": "Project 2",
            "title": "Explainability",
            "url_name": "project2:index",
            "description": "Explainability, counterfactuals, and feature effects.",
            "blurb": (
                "Decision tree vs logistic regression on Palmer Penguins, an "
                "accuracy–complexity λ trade-off, counterfactual explanations, and "
                "hand-written PDP / ALE feature-effect plots."
            ),
            "chips": ["λ tradeoff", "Counterfactuals", "PDP & ALE"],
        },
        {
            "name": "Project 3",
            "title": "Learning to Defer",
            "url_name": "project3:index",
            "description": "Learning to defer and active expert querying.",
            "blurb": (
                "A text classifier that can hand hard AG News articles to a simulated "
                "expert, with active learning to discover where the expert is strong. "
                "Includes a downloadable PDF report."
            ),
            "chips": ["Defer system", "Active learning", "Human expert"],
        },
        {
            "name": "Project 4",
            "title": "Preference Elicitation",
            "url_name": "project4:landing",
            "description": "Preference elicitation user study on movie data.",
            "blurb": (
                "A designed (not run) user study comparing pairwise choices with "
                "ranking for learning movie preferences, using a Bradley–Terry / "
                "Plackett–Luce model and a written PDF protocol."
            ),
            "chips": ["Bradley-Terry", "Plackett-Luce", "User study"],
        },
    ]

    return render(
        request,
        "home/index.html",
        {"students": students, "projects": projects},
    )

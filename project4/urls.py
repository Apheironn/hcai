from django.urls import path

from . import views

app_name = "project4"

urlpatterns = [
    path("", views.landing, name="landing"),
    path("study/", views.study, name="study"),
    path("results/", views.results, name="results"),
    path("report/", views.report, name="report"),
]

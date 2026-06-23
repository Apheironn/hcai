# HCAI Projects — Human-Centric Artificial Intelligence

Django web app with three course projects: supervised learning, explainability, and learning-to-defer with active learning.

**Author:** Mertcan Catak (642815) — solo project

## Setup

```bash
pip install -r requirements.txt
python manage.py runserver
```

Open http://127.0.0.1:8000/ (redirects to `/home/`).

**Note:** Project 3 loads the AG News dataset from Hugging Face on first visit (internet required).

## URLs

| Path | Description |
|------|-------------|
| `/home/` | Launch page and team info |
| `/project1/` | CSV upload, visualization, model training |
| `/project2/` | Palmer Penguins explainability (λ, counterfactuals, PDP/ALE) |
| `/project3/` | AG News deferral, active learning, PDF report, human expert UI |

## PDF task mapping

- **Project 1:** CSV upload & plots; train/test split; hyperparameter grid; metrics
- **Project 2:** Tree/logreg + λ slider; counterfactuals; custom PDP & ALE plots
- **Project 3:** Baseline classifier; simulated expert; learning-to-defer; active learning; optional human expert labeling; downloadable PDF report

## Design choices

- No Django ORM for ML state — sessions + files under `media/`
- Matplotlib plots saved to `media/plots/` and served as images
- Project 3 classifier/rejector cached per session in `media/model_cache/project3/`
- Human expert labels live in the session only (cleared when the session expires)

## Revert to pre-upgrade state

A snapshot commit and tag **`BEFORE`** marks the state before the quality upgrade:

```bash
git checkout BEFORE -- .
# or hard reset: git reset --hard BEFORE
```

## Tests

```bash
python manage.py test project1 project2 project3
```

# HCAI Projects — Human-Centric Artificial Intelligence

Django web app with four course projects: supervised learning, explainability,
learning-to-defer with active learning, and a preference elicitation user study.

**Author:** Mertcan Catak (642815) — solo project

## Setup

Requires Python 3.12.

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

Open http://127.0.0.1:8000/ (redirects to `/home/`).

**Note:** Project 3 loads the AG News dataset from Hugging Face on first visit
(internet required); the baseline classifier is then trained once and cached.
Project 4 reads `project4/data/movie_metadata.csv` (IMDB 5000 Movie Dataset),
which is included in the repository.

## URLs

| Path | Description |
|------|-------------|
| `/home/` | Launch page and team info |
| `/project1/` | CSV upload, visualization, model training |
| `/project2/` | Palmer Penguins explainability (λ, counterfactuals, PDP/ALE) |
| `/project3/` | AG News deferral, active learning, PDF report, human expert UI |
| `/project4/` | Preference elicitation study: landing page, PDF report, participant interface |

## PDF task mapping

- **Project 1:** CSV upload & plots; train/test split; hyperparameter grid; metrics
- **Project 2:** Tree/logreg + λ slider; counterfactuals; custom PDP & ALE plots
- **Project 3:** Baseline classifier; simulated expert; learning-to-defer; active learning; optional human expert labeling; downloadable PDF report
- **Project 4:** Movie feature representation (Task 1); Plackett-Luce ranking extension of Bradley-Terry (Task 2); user study design in the PDF report (Task 3); participant interface with both elicitation designs (Task 4)

## Design choices

- No Django ORM for ML state — sessions + files under `media/`
- Matplotlib plots saved to `media/plots/` and served as images
- Project 2's model bank and Project 3's baseline classifier are deterministic,
  so each is trained once per process and shared across sessions
  (`media/model_cache/**/shared/`)
- Human expert labels live in the session only (cleared when the session expires)
- Project 4 study responses (movies shown, submitted order, timings) live in the
  session only; the study design PDF is cached in `media/reports/`

## Tests

```bash
python manage.py test
```

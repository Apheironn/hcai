import os
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .dataset import GENRES, MIN_VOTES, MIN_YEAR, NUMERIC_FEATURES
from .study import HOLDOUT_PAIRS, PAIR_TASKS, RANK_SIZE, RANK_TASKS

FILENAME = "project4_study_design.pdf"


class StudyReport:
    """Static PDF describing the feature model, the ranking model and the study design."""

    LINE_WIDTH = 95

    def __init__(self, summary):
        self.summary = summary

    @classmethod
    def build(cls, summary, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        builder = cls(summary)
        with PdfPages(output_path) as pdf:
            for lines in builder._sections():
                builder._page(pdf, lines)
        return output_path

    def _sections(self):
        return [
            self._cover(),
            self._task1(),
            self._task2(),
            self._task3_design(),
            self._task3_procedure(),
            self._task3_analysis(),
            self._implementation(),
        ]

    # --- layout -------------------------------------------------------------------

    def _page(self, pdf, lines):
        for chunk in self._split_lines(lines):
            fig = plt.figure(figsize=(8.5, 11), facecolor="white")
            y = 0.94
            for line in chunk:
                if line.startswith("## "):
                    fig.text(0.08, y, line[3:], fontsize=13, weight="bold", family="sans-serif")
                    y -= 0.035
                elif line == "":
                    y -= 0.012
                else:
                    for part in textwrap.wrap(line, width=self.LINE_WIDTH) or [""]:
                        fig.text(0.08, y, part, fontsize=10, family="sans-serif", va="top")
                        y -= 0.028
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    def _split_lines(self, lines):
        pages, current, y = [], [], 0.94
        for line in lines:
            if line.startswith("## "):
                extra = 0.035
            elif line == "":
                extra = 0.012
            else:
                extra = 0.028 * max(1, len(textwrap.wrap(line, width=self.LINE_WIDTH)))
            if y - extra < 0.06 and current:
                pages.append(current)
                current, y = [], 0.94
            current.append(line)
            y -= extra
        if current:
            pages.append(current)
        return pages

    # --- content ------------------------------------------------------------------

    def _cover(self):
        s = self.summary
        return [
            "## Project 4 — Preference Elicitation for Movie Recommendation",
            "",
            "Course: Human-Centric Artificial Intelligence",
            "Author: Mertcan Catak (642815)",
            "Dataset: IMDB 5000 Movie Dataset (movie_metadata.csv)",
            "",
            f"Movies after filtering: {s['n_movies']}",
            f"Feature dimension: {s['n_features']} ({s['n_genres']} genres + {len(NUMERIC_FEATURES)} numeric)",
            f"Release years: {s['year_min']}–{s['year_max']}",
            "",
            "Contents:",
            "  1. Task 1 — feature representation and extraction",
            "  2. Task 2 — ranking extension of the Bradley-Terry model",
            "  3. Task 3 — design of the user study",
            "  4. Implementation notes for the interface (Task 4)",
            "",
            "The user study described here was designed but not conducted, as allowed by the",
            "project description. The interface is fully implemented and ready to collect data.",
        ]

    def _task1(self):
        s = self.summary
        return [
            "## Task 1 — Feature representation",
            "",
            "A movie is described by a vector x so that the utility U(x) = w^T x stays linear and",
            "interpretable: every coordinate of the estimated w can be read as 'how much this",
            "participant likes that property'. We therefore only keep attributes that a person can",
            "actually judge when looking at a movie card.",
            "",
            "Features (dimension {n}):".format(n=s["n_features"]),
            f"  • {s['n_genres']} binary genre indicators (multi-hot: a movie has several genres):",
            *textwrap.wrap(", ".join(GENRES) + ".", width=88, initial_indent="    ", subsequent_indent="    "),
            "  • IMDb rating (standardised): taste for critically appreciated movies.",
            "  • Runtime in minutes (standardised): tolerance for long movies.",
            "  • Release year (standardised): preference for recent versus older movies.",
            "  • log10(number of IMDb votes) (standardised): mainstream versus niche taste.",
            "",
            "Justification of the choices:",
            "  • Genre is the dominant driver of movie choice and is directly visible to the user,",
            "    so it forms the core of the representation.",
            "  • The four numeric features cover orthogonal dimensions (quality, length, era,",
            "    popularity) and keep the model small: with 19 parameters, a handful of comparisons",
            "    is already informative, which is exactly what fast elicitation requires.",
            "  • Rejected alternatives: budget and gross revenue (not visible to the user and",
            "    heavily missing), Facebook-like counts (noisy proxy of popularity, already covered",
            "    by the vote count), director and cast identity (thousands of sparse dummies would",
            "    make w impossible to estimate from ~10 interactions), and plot keyword text",
            "    embeddings (not interpretable in a linear utility of this size).",
            "",
            "Extraction method (project4/services/dataset.py):",
            "  1. Load movie_metadata.csv with pandas.",
            "  2. Drop rows missing title, genres, year, runtime, IMDb score or vote count.",
            f"  3. Keep movies with at least {MIN_VOTES} votes and released from {MIN_YEAR} on, and drop",
            "     duplicate title/year pairs. Rationale: participants can only express a preference",
            "     between movies they might plausibly recognise.",
            "  4. Split the pipe-separated genre string and build the multi-hot block in a fixed",
            "     genre order, so weights are comparable across sessions.",
            "  5. Standardise each numeric feature (z-score over the catalogue) and clip it to",
            "     [-3, 3], so that outliers such as 4-hour movies cannot dominate the utility.",
        ]

    def _task2(self):
        return [
            "## Task 2 — From pairwise Bradley-Terry to rankings",
            "",
            "Standard Bradley-Terry. For two movies i and j with utilities U(x_i) = w^T x_i,",
            "",
            "    P(i > j | w) = exp(w^T x_i) / ( exp(w^T x_i) + exp(w^T x_j) )",
            "                 = sigmoid( w^T (x_i - x_j) ).",
            "",
            "Design 2 collects a full ranking of ten movies, which is much more information than a",
            "single pair, so the likelihood has to be extended.",
            "",
            "Proposed extension (Plackett-Luce). We read a ranking i1 > i2 > ... > in as a sequence",
            "of independent choices: the participant first picks the best movie out of all n, then",
            "the best out of the remaining n-1, and so on. Each choice follows Bradley-Terry",
            "generalised to a set (a softmax over utilities), which gives",
            "",
            "    P(i1 > i2 > ... > in | w) = prod_{k=1..n-1}  exp(w^T x_{ik})",
            "                                             / sum_{j=k..n} exp(w^T x_{ij}).",
            "",
            "Justification:",
            "  • For n = 2 the product has a single factor and reduces exactly to Bradley-Terry, so",
            "    both interfaces are analysed with one common likelihood and the comparison between",
            "    Design 1 and Design 2 is not confounded by a change of model.",
            "  • It satisfies Luce's choice axiom: the probability of picking a movie from a set",
            "    depends only on the utilities in that set, which is the natural extension of the",
            "    pairwise assumption.",
            "  • It uses the full ranking, not only the winner, and it correctly weights the",
            "    information: the first positions are more informative than the last ones, where",
            "    fewer alternatives remain.",
            "  • Ranking a set of n items yields n(n-1)/2 implied pairwise judgements (45 for ten",
            "    movies) while remaining internally consistent (no cycles by construction).",
            "",
            "Estimation. The log-likelihood is concave in w, and we add a Gaussian prior",
            "N(0, 1/lambda) to keep the estimate finite when the observations are separable:",
            "",
            "    w_MAP = argmax_w  sum_over_rankings log P(ranking | w) - (lambda/2) ||w||^2.",
            "",
            "We maximise it with gradient ascent and a backtracking step size. The gradient of one",
            "ranking is sum_{k} ( x_{ik} - sum_{j>=k} p_j^{(k)} x_{ij} ), where p^{(k)} is the softmax",
            "over the utilities of the items still available at step k. With ~20 features and a few",
            "dozen rankings this converges in milliseconds, so the interface can re-estimate w after",
            "every interaction.",
            "",
            "Assumptions and limits: a single linear utility per user, no position or fatigue effect",
            "inside a ranking, and independence between the successive choices of one ranking. The",
            "study measures how well these assumptions hold in practice through held-out accuracy.",
        ]

    def _task3_design(self):
        return [
            "## Task 3 — User study: question, hypotheses and design",
            "",
            "Goal. Decide which of the two elicitation interfaces a cold-start movie recommender",
            "should use: repeated pairwise choices (Design 1) or ranking ten movies (Design 2).",
            "",
            "Research question. Which interface produces the more accurate preference vector w for",
            "a comparable amount of participant effort, and which one do participants prefer?",
            "",
            "Hypotheses:",
            "  • H1 (accuracy). Ranking ten movies yields a higher held-out prediction accuracy",
            "    than the same number of pairwise comparisons, because each ranking contributes",
            "    many implied comparisons.",
            "  • H2 (efficiency). Per second of participant time, Design 2 also yields higher",
            "    accuracy: the information gain outweighs the longer task duration.",
            "  • H3 (perceived effort). Design 1 is rated as easier and less mentally demanding,",
            "    because a single binary choice needs less working memory than ordering ten items.",
            "  • H0 for each: no difference between the two designs.",
            "",
            "Design. Within-subject, two blocks, one per interface. Every participant uses both",
            "interfaces, which removes between-person taste variance from the comparison and needs",
            "far fewer participants than a between-subject design.",
            "",
            "Counterbalancing and controls:",
            "  • Block order is counterbalanced across participants (half pairwise first, half",
            "    ranking first) to cancel learning and fatigue effects. In the interface the order",
            "    is fixed deterministically by the participant code, so the experimenter can keep",
            "    the two orders balanced from the recruitment list.",
            "  • Movies are drawn uniformly at random from the filtered catalogue, and all movies",
            "    shown to one participant are distinct, so no item is judged twice.",
            "  • Comparable workload per block: 10 pairwise comparisons versus 2 rankings of 10",
            "    movies. This equalises the number of movies inspected per block (20), which is the",
            "    fairest simple control; the resulting difference in implied comparisons (10 vs 90)",
            "    is exactly the effect H1 is about.",
            "",
            "Participants and recruitment:",
            "  • Target: 40 participants, students and staff recruited through university mailing",
            "    lists and course channels, plus snowball sampling.",
            "  • Inclusion: 18 or older, watches at least one movie per month, sufficient English",
            "    to read movie titles and genres.",
            "  • Sample size: a paired comparison with alpha = 0.05 and power 0.80 needs about 34",
            "    participants for a medium paired effect (d = 0.5); we recruit 40 to absorb",
            "    drop-outs and excluded sessions.",
            "  • Compensation: participation is voluntary, with a raffle of two cinema tickets.",
        ]

    def _task3_procedure(self):
        return [
            "## Task 3 — Procedure and measures",
            "",
            "Procedure (about 15 minutes, online, unsupervised):",
            "  1. Landing page: purpose of the study, what is recorded, duration, contact address.",
            "  2. Informed consent (explicit checkbox) and a short background form: participant",
            "     code, age group, movie-watching frequency.",
            f"  3. Block A instructions, then the {PAIR_TASKS} pairwise choices or {RANK_TASKS} rankings of {RANK_SIZE} movies.",
            "  4. Post-block questionnaire (mental effort, ease of expressing preferences,",
            "     confidence in the result).",
            "  5. Block B: the other interface, same task count, then the same questionnaire.",
            f"  6. Validation block: {HOLDOUT_PAIRS} pairwise choices on movies not seen before. These are the",
            "     held-out ground truth used to score both preference vectors, and they use the",
            "     same simple pairwise format for both designs so that neither is favoured.",
            "  7. Debriefing page: the estimated weights, the top recommendations of each model and",
            "     the measured times are shown to the participant.",
            "",
            "Dependent variables:",
            "  • Primary: accuracy of the estimated w on the held-out pairs, per design.",
            "  • Secondary: total and per-task completion time; accuracy per minute of elicitation;",
            "    log-likelihood of the held-out pairs (a graded version of accuracy).",
            "  • Subjective: three 5-point Likert items per design (effort, ease, confidence).",
            "  • Exploratory: learning curve of accuracy after each interaction, and agreement",
            "    between the two estimated weight vectors of the same participant (cosine",
            "    similarity), which measures whether both interfaces recover the same taste.",
            "",
            "Logged data: per task the movies shown, the submitted order, and the duration; per",
            "block the questionnaire answers. No names, no IP addresses; the participant code is",
            "the only identifier and is generated by the experimenter.",
            "",
            "Ethics and data protection: participation is voluntary and can be stopped at any time,",
            "data are pseudonymous and stored only in the study session, no sensitive category is",
            "collected, and the debriefing page explains what the model inferred. Movie titles and",
            "metadata come from a public dataset. The study would be submitted to the department",
            "ethics committee before recruitment.",
            "",
            "Pilot: 5 participants first, to check instruction clarity, the ranking interaction and",
            "the timing of the blocks. Pilot data are not part of the analysis.",
        ]

    def _task3_analysis(self):
        return [
            "## Task 3 — Analysis plan and validity",
            "",
            "Analysis:",
            "  1. Exclusion rules fixed in advance: sessions with a median pairwise decision time",
            "     below 1 second (random clicking) or an interrupted block are discarded.",
            "  2. For each participant and each design, fit w with the MAP estimator of Task 2 and",
            "     compute the accuracy on the held-out pairs.",
            "  3. H1 and H2: paired t-test on the per-participant difference (ranking minus",
            "     pairwise), or Wilcoxon signed-rank if the differences are not normal",
            "     (Shapiro-Wilk). Report the mean difference with a 95% confidence interval and",
            "     Cohen's d, not only the p-value.",
            "  4. H3: Wilcoxon signed-rank on the ordinal Likert items, reported per item.",
            "  5. Multiple comparisons: three confirmatory tests, Holm correction.",
            "  6. Exploratory analyses (learning curve, weight agreement, order effect as a",
            "     between-subject factor) are reported as descriptive results only.",
            "",
            "Expected outcome and decision rule. If Design 2 wins on accuracy and per-time accuracy",
            "but loses on perceived effort, the recommended product choice is a short ranking task",
            "for the very first session (cold start) followed by pairwise refinements, which stay",
            "cheap in attention. If the accuracy difference is not significant, the simpler and",
            "better-liked pairwise interface should be preferred.",
            "",
            "Threats to validity and mitigations:",
            "  • Unknown movies: a participant cannot rank movies they never saw. Mitigated by the",
            "    popularity filter, by an IMDb link on every card, and by an 'I don't know this",
            "    movie' report that flags the task for exclusion.",
            "  • Random items are often trivial to compare (very different movies), which inflates",
            "    accuracy for both designs. It affects both conditions equally; an adaptive",
            "    (information-maximising) item selection is the obvious follow-up study.",
            "  • Fatigue and learning across blocks: handled by counterbalancing and by keeping the",
            "    session short.",
            "  • Anchoring in Design 2: the list starts in a random order, so a participant who",
            "    reorders only a few movies leaves the rest in a random position rather than in a",
            "    systematically favourable one.",
            "  • Held-out set of six pairs is small, so accuracy is noisy per participant. The",
            "    paired design averages this noise over participants; the pilot is used to check",
            "    whether the validation block should be enlarged.",
            "  • Model mis-specification: if a linear utility over these 19 features cannot express",
            "    the participant's taste, both designs are penalised equally, but the absolute",
            "    accuracy level should be read with care.",
            "  • Preferences are not stable in time; the study only measures within-session",
            "    consistency, not long-term satisfaction with the recommendations.",
        ]

    def _implementation(self):
        return [
            "## Implementation notes (Task 4)",
            "",
            "The interface implements exactly the procedure above:",
            "  • Landing page: study description, this PDF, and the start form (participant code,",
            "    consent, background questions).",
            f"  • Design 1 screen: two movie cards side by side, one click per choice, {PAIR_TASKS} tasks.",
            f"  • Design 2 screen: the {RANK_SIZE} movies form a numbered list and the participant moves",
            "    them with up/down arrows until the list runs from most to least preferred. The",
            "    initial list order is random, which spreads any anchoring on the starting order",
            f"    evenly over the catalogue. {RANK_TASKS} tasks.",
            "  • Questionnaire screens after each block, validation block, and a debriefing page",
            "    with the estimated weights, both models' held-out accuracy and their top",
            "    recommendations.",
            "  • Every response is timed server-side and stored in the Django session; the",
            "    estimation runs on the Plackett-Luce MAP estimator described in Task 2.",
            "",
            "Files: project4/services/dataset.py (Task 1), project4/services/preference.py",
            "(Task 2), project4/services/study.py (study flow), project4/views.py and",
            "project4/templates/project4/ (interface), project4/services/report.py (this report).",
        ]

import random
import time
import zlib

from .preference import PreferenceModel

SESSION_KEY = "project4_study"

PAIRWISE = "pairwise"
RANKING = "ranking"

PAIR_TASKS = 10
RANK_TASKS = 2
RANK_SIZE = 10
HOLDOUT_PAIRS = 6

DESIGN_LABELS = {PAIRWISE: "Design 1 — pairwise choice", RANKING: "Design 2 — rank ten movies"}


class StudySession:
    """Session state machine for the two-block, within-subject elicitation study (Task 4)."""

    def __init__(self, data):
        self.data = data

    @classmethod
    def start(cls, request, dataset, participant):
        code = participant["code"]
        rng = random.Random(zlib.crc32(code.encode()))  # same code -> same movie sequence
        participant_code = code
        # Counterbalancing: the block order is fixed by the participant code parity.
        order = [PAIRWISE, RANKING] if sum(map(ord, participant_code)) % 2 == 0 else [RANKING, PAIRWISE]

        data = {
            "participant": participant,
            "order": order,
            "steps": cls._build_steps(dataset, order, rng),
            "step_index": 0,
            "step_started_at": time.time(),
            "responses": [],
            "surveys": {},
            "results": None,
        }
        request.session[SESSION_KEY] = data
        request.session.modified = True
        return cls(data)

    @classmethod
    def from_request(cls, request):
        data = request.session.get(SESSION_KEY)
        return cls(data) if data else None

    @staticmethod
    def _build_steps(dataset, order, rng):
        needed = PAIR_TASKS * 2 + RANK_TASKS * RANK_SIZE + HOLDOUT_PAIRS * 2
        pool = dataset.sample(needed, rng)  # distinct movies, so nothing is shown twice
        cursor = 0

        def take(n):
            nonlocal cursor
            chunk = pool[cursor : cursor + n]
            cursor += n
            return chunk

        steps = []
        for block, design in enumerate(order):
            steps.append({"kind": "intro", "design": design, "block": block})
            if design == PAIRWISE:
                for _ in range(PAIR_TASKS):
                    steps.append({"kind": "pair", "design": design, "block": block, "items": take(2)})
            else:
                for _ in range(RANK_TASKS):
                    steps.append({"kind": "rank", "design": design, "block": block, "items": take(RANK_SIZE)})
            steps.append({"kind": "survey", "design": design, "block": block})

        steps.append({"kind": "holdout_intro"})
        for _ in range(HOLDOUT_PAIRS):
            steps.append({"kind": "holdout", "design": "holdout", "items": take(2)})
        return steps

    def save(self, request):
        request.session[SESSION_KEY] = self.data
        request.session.modified = True

    @property
    def finished(self):
        return self.data["step_index"] >= len(self.data["steps"])

    def current(self):
        if self.finished:
            return None
        return self.data["steps"][self.data["step_index"]]

    def progress(self):
        total = sum(1 for step in self.data["steps"] if step["kind"] in {"pair", "rank", "holdout"})
        done = len(self.data["responses"])
        return done, total

    def mark_started(self):
        self.data["step_started_at"] = time.time()

    def _elapsed(self):
        return round(max(0.0, time.time() - self.data.get("step_started_at", time.time())), 2)

    def record_choice(self, ranking):
        step = self.current()
        if step is None or step["kind"] not in {"pair", "rank", "holdout"}:
            raise ValueError("No elicitation task is active.")
        if sorted(ranking) != sorted(step["items"]):
            raise ValueError("The submitted ranking does not match the movies shown.")
        self._append_response(step, ranking)

    def record_skip(self):
        """The participant does not know the movies shown; the task is logged but unused."""
        step = self.current()
        if step is None or step["kind"] not in {"pair", "rank", "holdout"}:
            raise ValueError("No elicitation task is active.")
        self._append_response(step, None)

    def _append_response(self, step, ranking):
        self.data["responses"].append(
            {
                "step": self.data["step_index"],
                "kind": step["kind"],
                "design": step["design"],
                "block": step.get("block"),
                "ranking": ranking,
                "skipped": ranking is None,
                "seconds": self._elapsed(),
            }
        )
        self.advance()

    def record_survey(self, answers):
        step = self.current()
        if step is None or step["kind"] != "survey":
            raise ValueError("No questionnaire is active.")
        self.data["surveys"][str(step["block"])] = {"design": step["design"], **answers}
        self.advance()

    def advance(self):
        self.data["step_index"] += 1
        self.mark_started()

    # --- analysis -----------------------------------------------------------------

    def build_results(self, dataset):
        answered = [r for r in self.data["responses"] if not r["skipped"]]
        holdout = [(r["ranking"][0], r["ranking"][1]) for r in answered if r["kind"] == "holdout"]

        designs = {}
        for design in self.data["order"]:
            responses = [r for r in answered if r["design"] == design]
            rankings = [r["ranking"] for r in responses]
            model = PreferenceModel.fit(dataset.features, rankings)
            implied = sum(len(PreferenceModel.implied_pairs(r)) for r in rankings)
            seconds = round(sum(r["seconds"] for r in responses), 1)
            designs[design] = {
                "label": DESIGN_LABELS[design],
                "n_tasks": len(responses),
                "n_implied_pairs": implied,
                "seconds": seconds,
                "seconds_per_task": round(seconds / len(responses), 1) if responses else None,
                "holdout_accuracy": model.pair_accuracy(dataset.features, holdout),
                "weights": [round(float(w), 4) for w in model.weights],
                "top_weights": model.weight_table(dataset.feature_names),
                "recommendations": dataset.cards(model.recommend(dataset.features, k=3)),
            }

        results = {
            "designs": designs,
            "order": self.data["order"],
            "n_holdout": len(holdout),
            "surveys": self.data["surveys"],
            "feature_names": dataset.feature_names,
        }
        self.data["results"] = results
        return results

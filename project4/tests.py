import os
from unittest.mock import patch

import numpy as np
from django.test import TestCase

from project4.services.preference import PreferenceModel
from project4.services.study import SESSION_KEY

START_DATA = {
    "action": "start",
    "participant_code": "P01",
    "age_group": "25-34",
    "movie_frequency": "monthly",
    "consent": "on",
}


class PreferenceModelTests(TestCase):
    def test_pairwise_fit_recovers_preference_direction(self):
        features = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
        # The synthetic user always prefers feature 0 over feature 1.
        rankings = [[0, 1], [2, 1], [0, 3], [2, 3]]
        model = PreferenceModel.fit(features, rankings)

        self.assertGreater(model.weights[0], 0)
        self.assertEqual(model.pair_accuracy(features, [(0, 1)]), 1.0)

    def test_ranking_likelihood_extends_pairwise(self):
        features = np.eye(4)
        model = PreferenceModel.fit(features, [[0, 1, 2, 3]])
        utilities = model.utilities(features)

        self.assertTrue(np.all(np.diff(utilities) < 0))
        self.assertEqual(len(PreferenceModel.implied_pairs([0, 1, 2, 3])), 6)


class StudyFlowTests(TestCase):
    def test_landing_page(self):
        response = self.client.get("/project4/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Preference Elicitation")

    def _answer_all_steps(self):
        self.client.post("/project4/", START_DATA)
        for _ in range(200):
            data = self.client.session.get(SESSION_KEY)
            if data["step_index"] >= len(data["steps"]):
                return
            step = data["steps"][data["step_index"]]
            if step["kind"] in {"intro", "holdout_intro"}:
                payload = {"action": "continue"}
            elif step["kind"] in {"pair", "holdout"}:
                payload = {"action": "choose", "choice": step["items"][0]}
            elif step["kind"] == "rank":
                payload = {"action": "rank", "order": ",".join(str(i) for i in step["items"])}
            else:
                payload = {"action": "survey", "effort": "3", "ease": "4", "confidence": "4"}
            self.client.post("/project4/study/", payload)
        self.fail("The study did not finish.")

    @patch("project4.views.StudyPlots.design_comparison", return_value="/media/plots/comparison.png")
    @patch("project4.views.StudyPlots.learned_weights", return_value="/media/plots/weights.png")
    def test_complete_study_produces_results(self, _weights, _comparison):
        self._answer_all_steps()

        response = self.client.get("/project4/results/")
        self.assertEqual(response.status_code, 200)

        results = self.client.session[SESSION_KEY]["results"]
        self.assertEqual(sorted(results["designs"]), ["pairwise", "ranking"])
        self.assertEqual(len(results["surveys"]), 2)
        for design in results["designs"].values():
            self.assertIsNotNone(design["holdout_accuracy"])

    def test_study_redirects_without_session(self):
        self.assertRedirects(self.client.get("/project4/study/"), "/project4/")

    @patch("project4.views.StudyReport.build")
    def test_report_download(self, mock_build):
        def write_pdf(summary, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(b"%PDF-1.4")

        mock_build.side_effect = write_pdf
        response = self.client.get("/project4/report/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")

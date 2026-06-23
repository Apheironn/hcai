from unittest.mock import MagicMock, patch

import os

from django.test import TestCase


class Project3Tests(TestCase):
    @patch("project3.views.ResultPlots.class_distribution", return_value="/media/plots/test.png")
    @patch("project3.views.ModelStore.ensure")
    @patch("project3.views.AgNewsDataset.load")
    def test_index_loads(self, mock_load, mock_ensure, _mock_plot):
        dataset = MagicMock()
        dataset.n_train = 120000
        dataset.n_test = 7600
        dataset.preview.return_value = "<table></table>"
        dataset.class_counts.return_value = {"World": 1, "Sports": 1, "Business": 1, "Sci/Tech": 1}
        mock_load.return_value = dataset

        store = MagicMock()
        store.data = {
            "baseline_accuracy": 0.91,
            "human_labels": {},
            "human_batch_indices": [],
        }
        store.expert = MagicMock()
        mock_ensure.return_value = store

        response = self.client.get("/project3/")
        self.assertEqual(response.status_code, 200)

    @patch("project3.views.ResultPlots.human_vs_simulated", return_value="/media/plots/human.png")
    @patch("project3.views.HumanExpertSession.build_report")
    @patch("project3.views.ResultPlots.class_distribution", return_value="/media/plots/test.png")
    @patch("project3.views.ModelStore.ensure")
    @patch("project3.views.AgNewsDataset.load")
    def test_label_expert(self, mock_load, mock_ensure, _mock_plot, mock_report, _mock_chart):
        dataset = MagicMock()
        dataset.n_train = 100
        dataset.train_texts = ["news"] * 100
        dataset.train_labels = [0] * 100
        dataset.preview.return_value = "<table></table>"
        dataset.class_counts.return_value = {"World": 1, "Sports": 1, "Business": 1, "Sci/Tech": 1}
        mock_load.return_value = dataset

        store = MagicMock()
        store.data = {
            "baseline_accuracy": 0.91,
            "human_labels": {},
            "human_batch_indices": [0],
        }
        store.expert = MagicMock()

        def save_side_effect(request):
            store.data.setdefault("human_labels", {})["0"] = 1

        store.save.side_effect = save_side_effect
        mock_ensure.return_value = store
        mock_report.return_value = {
            "n_labeled": 1,
            "human_accuracy": 1.0,
            "expert_accuracy": 0.5,
            "per_class": {"World": {"human": 0.0, "expert": 0.0}, "Sports": {"human": None, "expert": None}, "Business": {"human": None, "expert": None}, "Sci/Tech": {"human": None, "expert": None}},
        }

        self.client.post(
            "/project3/",
            {"action": "label_expert", "article_index": 0, "class_name": "Sports"},
        )
        response = self.client.post("/project3/", {"action": "finish_human"})
        self.assertEqual(response.status_code, 200)

    @patch("project3.views.ReportBuilder.build")
    @patch("project3.views.ModelStore.from_session")
    @patch("project3.views.AgNewsDataset.load")
    def test_report_download(self, mock_load, mock_from_session, mock_build):
        mock_load.return_value = MagicMock()
        mock_from_session.return_value = MagicMock(data={"baseline_accuracy": 0.91})

        def write_report(dataset, data, path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as handle:
                handle.write(b"%PDF-1.4")

        mock_build.side_effect = write_report
        session = self.client.session
        session["project3_store"] = {"baseline_accuracy": 0.91, "classifier_path": "x"}
        session.save()

        response = self.client.get("/project3/report/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")

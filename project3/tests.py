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
        store.data = {"baseline_accuracy": 0.91}
        mock_ensure.return_value = store

        response = self.client.get("/project3/")
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
        session["project3_store"] = {"baseline_accuracy": 0.91, "classifier_path": "x", "baseline_accuracy": 0.91}
        session.save()

        response = self.client.get("/project3/report/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")

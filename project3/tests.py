from unittest.mock import MagicMock, patch

from django.test import TestCase


class Project3Tests(TestCase):
    @patch("project3.views.ModelStore.ensure")
    @patch("project3.views.AgNewsDataset.load")
    def test_index_loads(self, mock_load, mock_ensure):
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

import os
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings


SAMPLE_CSV = b"""SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm,Species
5.1,3.5,1.4,0.2,1
4.9,3.0,1.4,0.2,1
7.0,3.2,4.7,1.4,2
6.4,3.2,4.5,1.5,2
6.3,3.3,6.0,2.5,3
5.8,2.7,5.1,1.9,3
"""


@override_settings(MEDIA_ROOT=tempfile.mkdtemp())
class Project1Tests(TestCase):
    def test_index_loads(self):
        response = self.client.get("/project1/")
        self.assertEqual(response.status_code, 200)

    def test_load_sample(self):
        response = self.client.post("/project1/", {"action": "load_sample"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dataset summary")

    def test_upload_and_train(self):
        self.client.post("/project1/", {"action": "load_sample"})
        response = self.client.post(
            "/project1/",
            {
                "action": "train",
                "model": "DecisionTreeClassifier",
                "test_size": 60,
                "param_values": "2,5",
                "metric": "accuracy",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "results-table")

from django.test import TestCase


class Project3Tests(TestCase):
    def test_index_loads(self):
        response = self.client.get("/project3/")
        self.assertEqual(response.status_code, 200)

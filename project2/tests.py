from django.test import TestCase


class Project2Tests(TestCase):
    def test_index_loads(self):
        response = self.client.get("/project2/")
        self.assertEqual(response.status_code, 200)

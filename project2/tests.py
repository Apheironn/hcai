from django.test import TestCase


class Project2Tests(TestCase):
    def test_index_loads(self):
        response = self.client.get("/project2/")
        self.assertEqual(response.status_code, 200)

    def test_select_json(self):
        session = self.client.session
        session.save()
        response = self.client.get("/project2/select/?model_type=logreg&lambda=0.01")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("candidates", data)
        self.assertIn("pareto_url", data)

    def test_preview_json(self):
        session = self.client.session
        session.save()
        self.client.get("/project2/")
        response = self.client.get("/project2/preview/?model_type=tree&lambda=0&row_index=0")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("label", data)
        self.assertIn("predicted", data)

    def test_counterfactual_post(self):
        self.client.get("/project2/")
        response = self.client.post(
            "/project2/",
            {
                "action": "counterfactual",
                "model_type": "tree",
                "lambda_value": 0.0,
                "row_index": 0,
                "target_class": "Gentoo",
            },
        )
        self.assertEqual(response.status_code, 200)

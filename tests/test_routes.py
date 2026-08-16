import unittest
import json
from app import app

class TestRoutes(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        self.client = app.test_client()

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Car", response.data)
        self.assertIn(b"Sence", response.data)

    def test_health_route(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_api_model_info_route(self):
        response = self.client.get('/api/model-info')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["n_features"], 19)

    def test_predict_form_post_success(self):
        payload = {
            "year": "2019",
            "km_driven": "45000",
            "transmission": "Automatic",
            "mileage": "18.5",
            "seats": "5",
            "engine": "1998",
            "max_power": "150",
            "fuel": "Petrol",
            "seller_type": "Individual",
            "owner": "First Owner",
            "brand": "BMW"
        }
        response = self.client.post('/predict', data=payload)
        self.assertEqual(response.status_code, 200)

    def test_predict_form_post_invalid(self):
        invalid_payload = {
            "year": "1850",  # Invalid year
            "km_driven": "-500",
            "transmission": "InvalidTrans",
            "brand": ""
        }
        response = self.client.post('/predict', data=invalid_payload)
        self.assertEqual(response.status_code, 400)

    def test_api_predict_json_success(self):
        payload = {
            "year": 2019,
            "km_driven": 45000,
            "transmission": "Automatic",
            "mileage": 18.5,
            "seats": 5,
            "engine": 1998,
            "max_power": 150,
            "fuel": "Petrol",
            "seller_type": "Individual",
            "owner": "First Owner",
            "brand": "BMW"
        }
        response = self.client.post(
            '/api/predict',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data["success"])
        self.assertGreater(data["predicted_price"], 0)
        self.assertIn("formatted_inr", data)
        self.assertIn("formatted_lakhs", data)

if __name__ == '__main__':
    unittest.main()

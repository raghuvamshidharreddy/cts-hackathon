import unittest
from config import Config
from services.prediction import ModelService, format_inr

class TestPredictionService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.service = ModelService(Config.MODEL_PATH)

    def test_model_loaded(self):
        self.assertTrue(self.service.is_loaded)
        self.assertIsNotNone(self.service.model)

    def test_model_feature_contract(self):
        info = self.service.get_model_info()
        self.assertEqual(info["n_features"], 19)
        self.assertEqual(info["expected_features"], Config.EXPECTED_FEATURES)

    def test_predict_valid_sample(self):
        sample_data = {
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
        res = self.service.predict(sample_data)
        self.assertTrue(res["success"])
        self.assertGreater(res["raw_price"], 0)
        self.assertTrue(res["price_inr"].startswith("₹"))
        self.assertTrue(res["price_lakhs"].endswith("Lakhs"))
        self.assertGreater(len(res["top_features"]), 0)

    def test_format_inr(self):
        self.assertEqual(format_inr(645000), "₹6,45,000")
        self.assertEqual(format_inr(875000), "₹8,75,000")
        self.assertEqual(format_inr(12500000), "₹1,25,00,000")

if __name__ == '__main__':
    unittest.main()

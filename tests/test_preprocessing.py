import unittest
import numpy as np
import pandas as pd
from config import Config
from services.preprocessing import transform_input

class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.sample_data = {
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

    def test_feature_count_and_order(self):
        df = transform_input(self.sample_data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.shape[1], 19)
        self.assertEqual(list(df.columns), Config.EXPECTED_FEATURES)

    def test_log_transformations(self):
        df = transform_input(self.sample_data)
        expected_engine_log = np.log(1998)
        expected_power_log = np.log1p(150)
        self.assertAlmostEqual(df["engine_log"].iloc[0], expected_engine_log, places=5)
        self.assertAlmostEqual(df["max_power_log"].iloc[0], expected_power_log, places=5)

    def test_luxury_brand(self):
        df_bmw = transform_input(self.sample_data)
        self.assertEqual(df_bmw["is_luxury"].iloc[0], 1)

        data_maruti = self.sample_data.copy()
        data_maruti["brand"] = "Maruti"
        df_maruti = transform_input(data_maruti)
        self.assertEqual(df_maruti["is_luxury"].iloc[0], 0)

    def test_derived_features(self):
        df = transform_input(self.sample_data)
        engine_log = np.log(1998)
        max_power_log = np.log1p(150)

        # Luxury engine
        self.assertAlmostEqual(df["luxury_engine"].iloc[0], engine_log, places=5)

        # Performance index
        self.assertAlmostEqual(df["performance_index"].iloc[0], max_power_log * engine_log, places=5)

        # Tech score
        self.assertAlmostEqual(df["tech_score"].iloc[0], max_power_log - engine_log, places=5)

        # Petrol wear
        self.assertEqual(df["petrol_wear"].iloc[0], 45000)

        # Age wear with reference year 2025
        expected_age_wear = (2025 - 2019) * 45000
        self.assertEqual(df["age_wear"].iloc[0], expected_age_wear)

if __name__ == '__main__':
    unittest.main()

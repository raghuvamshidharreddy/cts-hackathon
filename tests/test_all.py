"""
Tests — Car Sence v2 (Lasso Regression Edition)
Covers preprocessing, prediction service, and Flask routes.
"""
import unittest
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ALL_FEATURES, BRAND_MODELS, MODEL_PATH
from services.preprocessing import build_feature_vector
from services.prediction import ModelService
from services.validation import validate_input


# ─────────────────────────────────────────────────────────────────────
class TestPreprocessing(unittest.TestCase):

    def _base_input(self, **overrides):
        data = dict(brand='Toyota', model='Camry', year=2018, engine_size=2.0,
                    mileage=45000, fuel_type='Petrol', transmission='Automatic',
                    doors=4, owner_count=1)
        data.update(overrides)
        return data

    def test_output_shape(self):
        df = build_feature_vector(self._base_input())
        self.assertEqual(df.shape, (1, 59), "Feature vector must have shape (1, 59)")

    def test_column_order(self):
        df = build_feature_vector(self._base_input())
        self.assertEqual(list(df.columns), ALL_FEATURES)

    def test_brand_toyota_one_hot(self):
        df = build_feature_vector(self._base_input(brand='Toyota'))
        self.assertEqual(df['Brand_Toyota'].iloc[0], 1)
        self.assertEqual(df['Brand_BMW'].iloc[0], 0)

    def test_brand_audi_baseline(self):
        """Audi is the baseline brand — all Brand_* flags must be 0."""
        df = build_feature_vector(self._base_input(brand='Audi', model='A4'))
        for col in [c for c in ALL_FEATURES if c.startswith('Brand_')]:
            self.assertEqual(df[col].iloc[0], 0, f"{col} should be 0 for Audi baseline")

    def test_bmw_3_series_baseline_model(self):
        """BMW 3 Series is the drop_first model baseline — all Model_* flags must be 0."""
        df = build_feature_vector(self._base_input(brand='BMW', model='3 Series'))
        for col in [c for c in ALL_FEATURES if c.startswith('Model_')]:
            self.assertEqual(df[col].iloc[0], 0, f"{col} should be 0 for 3 Series baseline")

    def test_model_camry(self):
        df = build_feature_vector(self._base_input(brand='Toyota', model='Camry'))
        self.assertEqual(df['Model_Camry'].iloc[0], 1)

    def test_fuel_diesel_baseline(self):
        """Diesel is baseline — all Fuel_Type_* flags should be 0."""
        df = build_feature_vector(self._base_input(fuel_type='Diesel'))
        self.assertEqual(df['Fuel_Type_Petrol'].iloc[0], 0)
        self.assertEqual(df['Fuel_Type_Electric'].iloc[0], 0)
        self.assertEqual(df['Fuel_Type_Hybrid'].iloc[0], 0)

    def test_fuel_electric(self):
        df = build_feature_vector(self._base_input(fuel_type='Electric'))
        self.assertEqual(df['Fuel_Type_Electric'].iloc[0], 1)

    def test_transmission_automatic_baseline(self):
        df = build_feature_vector(self._base_input(transmission='Automatic'))
        self.assertEqual(df['Transmission_Manual'].iloc[0], 0)
        self.assertEqual(df['Transmission_Semi-Automatic'].iloc[0], 0)

    def test_transmission_manual(self):
        df = build_feature_vector(self._base_input(transmission='Manual'))
        self.assertEqual(df['Transmission_Manual'].iloc[0], 1)

    def test_doors_4(self):
        df = build_feature_vector(self._base_input(doors=4))
        self.assertEqual(df['Doors_4'].iloc[0], 1)
        self.assertEqual(df['Doors_3'].iloc[0], 0)
        self.assertEqual(df['Doors_5'].iloc[0], 0)

    def test_owner_1_baseline(self):
        df = build_feature_vector(self._base_input(owner_count=1))
        for col in [c for c in ALL_FEATURES if c.startswith('Owner_Count_')]:
            self.assertEqual(df[col].iloc[0], 0, f"{col} should be 0 for 1 owner")

    def test_owner_2(self):
        df = build_feature_vector(self._base_input(owner_count=2))
        self.assertEqual(df['Owner_Count_2'].iloc[0], 1)

    def test_engine_small_baseline(self):
        """Engine < 1.6L = Small = baseline, all Engine_Size_Group_* = 0."""
        df = build_feature_vector(self._base_input(engine_size=1.2))
        self.assertEqual(df['Engine_Size_Group_Medium'].iloc[0], 0)
        self.assertEqual(df['Engine_Size_Group_Large'].iloc[0], 0)

    def test_engine_medium(self):
        df = build_feature_vector(self._base_input(engine_size=2.0))
        self.assertEqual(df['Engine_Size_Group_Medium'].iloc[0], 1)
        self.assertEqual(df['Engine_Size_Group_Large'].iloc[0], 0)

    def test_engine_large(self):
        df = build_feature_vector(self._base_input(engine_size=3.5))
        self.assertEqual(df['Engine_Size_Group_Large'].iloc[0], 1)

    def test_mileage_low_baseline(self):
        """Mileage < 30,000 = Low = baseline, all Mileage_Group_* = 0."""
        df = build_feature_vector(self._base_input(mileage=20000))
        self.assertEqual(df['Mileage_Group_Medium'].iloc[0], 0)
        self.assertEqual(df['Mileage_Group_High'].iloc[0], 0)

    def test_mileage_medium(self):
        df = build_feature_vector(self._base_input(mileage=50000))
        self.assertEqual(df['Mileage_Group_Medium'].iloc[0], 1)

    def test_mileage_high(self):
        df = build_feature_vector(self._base_input(mileage=90000))
        self.assertEqual(df['Mileage_Group_High'].iloc[0], 1)

    def test_year_2010s(self):
        df = build_feature_vector(self._base_input(year=2015))
        self.assertEqual(df['Year_of_Registration_Group_2010s'].iloc[0], 1)
        self.assertEqual(df['Year_of_Registration_Group_2020s'].iloc[0], 0)

    def test_year_2020s(self):
        df = build_feature_vector(self._base_input(year=2022))
        self.assertEqual(df['Year_of_Registration_Group_2020s'].iloc[0], 1)

    def test_year_2000s_baseline(self):
        df = build_feature_vector(self._base_input(year=2005))
        self.assertEqual(df['Year_of_Registration_Group_2010s'].iloc[0], 0)
        self.assertEqual(df['Year_of_Registration_Group_2020s'].iloc[0], 0)

    def test_numeric_features(self):
        df = build_feature_vector(self._base_input(year=2018, engine_size=2.0, mileage=45000))
        self.assertEqual(df['Year'].iloc[0], 2018)
        self.assertAlmostEqual(df['Engine_Size'].iloc[0], 2.0, places=4)
        self.assertAlmostEqual(df['Mileage'].iloc[0], 45000, places=0)


# ─────────────────────────────────────────────────────────────────────
class TestPredictionService(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.svc = ModelService()
        cls.svc.load(MODEL_PATH)

    def _predict(self, **overrides):
        data = dict(brand='Toyota', model='Camry', year=2018, engine_size=2.0,
                    mileage=45000, fuel_type='Petrol', transmission='Automatic',
                    doors=4, owner_count=1)
        data.update(overrides)
        return self.svc.predict(data)

    def test_model_loaded(self):
        self.assertTrue(self.svc.is_loaded)

    def test_returns_success(self):
        r = self._predict()
        self.assertTrue(r['success'])

    def test_price_positive(self):
        r = self._predict()
        self.assertGreater(r['predicted_price'], 0)

    def test_usd_format(self):
        r = self._predict()
        self.assertIn('$', r['formatted_usd'])

    def test_price_range_present(self):
        r = self._predict()
        self.assertIn('price_low', r)
        self.assertIn('price_high', r)

    def test_range_low_lt_high(self):
        r = self._predict()
        # Strip $ and , then compare
        low  = float(r['price_low'].replace('$','').replace(',',''))
        high = float(r['price_high'].replace('$','').replace(',',''))
        self.assertLess(low, high)

    def test_mileage_monotonicity(self):
        """Higher mileage → lower price (more wear)."""
        r_low  = self._predict(mileage=10000)
        r_high = self._predict(mileage=100000)
        self.assertGreater(
            r_low['predicted_price'], r_high['predicted_price'],
            "Low mileage car should be priced higher than high mileage car"
        )

    def test_mileage_monotonicity_ordered(self):
        """p_10k > p_40k > p_80k > p_150k — strictly ordered."""
        p10  = self._predict(mileage=10000)['predicted_price']
        p40  = self._predict(mileage=40000)['predicted_price']
        p80  = self._predict(mileage=80000)['predicted_price']
        p150 = self._predict(mileage=150000)['predicted_price']
        self.assertGreater(p10, p40, "10k miles should cost more than 40k miles")
        self.assertGreater(p40, p80, "40k miles should cost more than 80k miles")
        self.assertGreater(p80, p150, "80k miles should cost more than 150k miles")

    def test_newer_car_costs_more(self):
        """Newer year (2022) should cost more than older year (2010)."""
        r_new = self._predict(year=2022)
        r_old = self._predict(year=2010)
        self.assertGreater(r_new['predicted_price'], r_old['predicted_price'])

    def test_feature_names(self):
        names = self.svc.feature_names()
        self.assertEqual(len(names), 59)

    def test_bmw_premium_over_kia(self):
        """BMW X5 should generally be priced higher than Kia Rio."""
        r_bmw = self._predict(brand='BMW', model='X5', engine_size=3.0)
        r_kia = self._predict(brand='Kia', model='Rio', engine_size=1.4)
        self.assertGreater(r_bmw['predicted_price'], r_kia['predicted_price'])

    def test_bmw_3_series_returns_prediction(self):
        """BMW 3 Series (baseline model, all Model_*=0) should return valid price."""
        r = self._predict(brand='BMW', model='3 Series', engine_size=2.0)
        self.assertTrue(r['success'])
        self.assertGreater(r['predicted_price'], 0)


# ─────────────────────────────────────────────────────────────────────
class TestValidation(unittest.TestCase):

    def _base(self, **overrides):
        data = dict(brand='Toyota', model='Camry', year=2018, engine_size=2.0,
                    mileage=45000, fuel_type='Petrol', transmission='Automatic',
                    doors=4, owner_count=1)
        data.update(overrides)
        return data

    def test_valid_passes(self):
        r = validate_input(self._base())
        self.assertTrue(r['valid'])

    def test_year_2000_valid(self):
        r = validate_input(self._base(year=2000))
        self.assertTrue(r['valid'])

    def test_year_2023_valid(self):
        r = validate_input(self._base(year=2023))
        self.assertTrue(r['valid'])

    def test_missing_brand_fails(self):
        r = validate_input(self._base(brand=''))
        self.assertFalse(r['valid'])

    def test_bad_brand_fails(self):
        r = validate_input(self._base(brand='Ferrari'))
        self.assertFalse(r['valid'])

    def test_bad_year_fails(self):
        r = validate_input(self._base(year=1800))
        self.assertFalse(r['valid'])

    def test_bad_engine_fails(self):
        r = validate_input(self._base(engine_size=-1))
        self.assertFalse(r['valid'])

    def test_bad_mileage_fails(self):
        r = validate_input(self._base(mileage=-500))
        self.assertFalse(r['valid'])

    def test_bad_fuel_fails(self):
        r = validate_input(self._base(fuel_type='CNG'))
        self.assertFalse(r['valid'])

    def test_bad_transmission_fails(self):
        r = validate_input(self._base(transmission='CVT'))
        self.assertFalse(r['valid'])

    def test_bad_doors_fails(self):
        r = validate_input(self._base(doors=7))
        self.assertFalse(r['valid'])

    def test_bad_owners_fails(self):
        r = validate_input(self._base(owner_count=10))
        self.assertFalse(r['valid'])

    def test_audi_valid(self):
        r = validate_input(self._base(brand='Audi', model='A4'))
        self.assertTrue(r['valid'])


# ─────────────────────────────────────────────────────────────────────
class TestFlaskRoutes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import app as application
        application.app.config['TESTING'] = True
        cls.client = application.app.test_client()

    def test_index_200(self):
        r = self.client.get('/')
        self.assertEqual(r.status_code, 200)

    def test_index_contains_brand(self):
        r = self.client.get('/')
        self.assertIn(b'Toyota', r.data)

    def test_health_ok(self):
        r = self.client.get('/health')
        d = json.loads(r.data)
        self.assertEqual(d['status'], 'ok')
        self.assertTrue(d['model_loaded'])

    def test_model_info(self):
        r = self.client.get('/api/model-info')
        d = json.loads(r.data)
        self.assertEqual(d['total_features'], 59)
        self.assertEqual(d['currency'], 'USD')

    def test_predict_valid(self):
        payload = dict(brand='Toyota', model='Camry', year=2018, engine_size=2.0,
                       mileage=45000, fuel_type='Petrol', transmission='Automatic',
                       doors=4, owner_count=1)
        r = self.client.post('/api/predict',
                             data=json.dumps(payload),
                             content_type='application/json')
        self.assertEqual(r.status_code, 200)
        d = json.loads(r.data)
        self.assertTrue(d['success'])
        self.assertIn('formatted_usd', d)

    def test_predict_invalid_brand(self):
        payload = dict(brand='Ferrari', year=2018, engine_size=2.0,
                       mileage=45000, fuel_type='Petrol', transmission='Automatic',
                       doors=4, owner_count=1)
        r = self.client.post('/api/predict',
                             data=json.dumps(payload),
                             content_type='application/json')
        self.assertEqual(r.status_code, 400)

    def test_404(self):
        r = self.client.get('/nonexistent-route')
        self.assertEqual(r.status_code, 404)


if __name__ == '__main__':
    unittest.main(verbosity=2)

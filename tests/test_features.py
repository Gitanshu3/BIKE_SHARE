import unittest
import pandas as pd
from bike_sharing_model.processing.features import (
    WeekdayImputer,
    WeathersitImputer,
    Mapper,
    OutlierHandler,
    WeekdayOneHotEncoder
)

class TestFeatures(unittest.TestCase):

    def test_weekday_imputer(self):
        df = pd.DataFrame({
            'dteday': ['2024-02-25', '2024-02-26'],
            'weekday': [None, 'Mon']
        })
        transformer = WeekdayImputer()
        transformed_df = transformer.transform(df)
        self.assertEqual(transformed_df.loc[0, 'weekday'], 'Sun')
        self.assertEqual(transformed_df.loc[1, 'weekday'], 'Mon')

    def test_weathersit_imputer(self):
        df = pd.DataFrame({'weathersit': [1, None, 2, None, 1]})
        transformer = WeathersitImputer()
        transformer.fit(df)
        transformed_df = transformer.transform(df)
        self.assertEqual(transformed_df['weathersit'].isna().sum(), 0)
        self.assertEqual(transformed_df['weathersit'].iloc[1], 1)

    def test_mapper(self):
        df = pd.DataFrame({'season': [1, 2, 3, 4]})
        mappings = {'season': {1: 'Winter', 2: 'Spring', 3: 'Summer', 4: 'Fall'}}
        transformer = Mapper(variables=['season'], mappings=mappings)
        transformed_df = transformer.transform(df)
        self.assertEqual(list(transformed_df['season']), ['Winter', 'Spring', 'Summer', 'Fall'])

    def test_outlier_handler(self):
        df = pd.DataFrame({'temp': [0.1, 0.5, 0.9, 1.5, -0.5]})
        transformer = OutlierHandler(variables=['temp'])
        transformer.fit(df)
        transformed_df = transformer.transform(df)
        self.assertTrue(transformed_df['temp'].between(transformer.lower_bounds['temp'], transformer.upper_bounds['temp']).all())

    def test_weekday_one_hot_encoder(self):
        df = pd.DataFrame({'weekday': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']})
        transformer = WeekdayOneHotEncoder()
        transformed_df = transformer.transform(df)
        expected_cols = ['weekday_Fri', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun', 'weekday_Thu', 'weekday_Tue', 'weekday_Wed']
        self.assertTrue(all(col in transformed_df.columns for col in expected_cols))
        self.assertEqual(transformed_df.sum().sum(), 7)

if __name__ == '__main__':
    unittest.main()

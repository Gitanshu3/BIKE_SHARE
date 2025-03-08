import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from bike_sharing_model import predict
from sklearn.metrics import r2_score, mean_squared_error


class TestPrediction(unittest.TestCase):

    def test_make_prediction_valid_input(self):

        data_in = [{'dteday': '2011-07-13', 'season': 'fall', 'hr': '4am', 'holiday': 'No', 'weekday': 'Wed',
                    'workingday': 'Yes', 'weathersit': 'Clear', 'temp': 26.78, 'atemp': 28.9988, 'hum': 58.0,
                    'windspeed': 16.9979, 'casual': 0, 'registered': 5, 'year':2012, 'month':4},

                   {'dteday': '2007-07-13', 'season': 'Spring', 'hr': '9pm', 'holiday': 'Yes',
            'weekday': 'Sun', 'workingday': 'No', 'weathersit': 'Clear',
            'temp': 96.78, 'atemp': 8.9988, 'hum': 8.0, 'windspeed': 116.9979,
            'casual': 0, 'registered': 2, 'year': 2019, 'month': 9
        },
                   {
                       'dteday': '2023-07-13', 'season': 'summer','hr': '8pm','holiday': 'No','weekday': 'Sun','workingday': 'No','weathersit': 'Light Rain','temp': 8.92,
                       'atemp': 5.9978,'hum': 93.0, 'windspeed': 27.9993,'casual': 1,'registered': 34,'year': 2012,'month': 4
                   }]

        results = predict.make_prediction(input_data=data_in)
        self.assertIn("predictions", results)
        # self.assertEqual(results["predictions"][0], 5.0)
        self.assertIn("version", results)

        expected_predictions = [5.0, 2.0, 35.0]

        r2 = r2_score(expected_predictions, results["predictions"])
        print(r2)
        assert r2 > 0.8

        mse = mean_squared_error(expected_predictions, results["predictions"])
        print(mse)
        assert mse < 0.5



if __name__ == "__main__":
    unittest.main()
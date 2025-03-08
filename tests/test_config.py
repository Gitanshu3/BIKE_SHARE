import unittest
from pathlib import Path
from bike_sharing_model.config import core


class TestConfig(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config_path = Path(__file__).parent / "configTest.yml"

    def test_create_and_validate_config(self):
        parsed_yaml = core.fetch_config_from_yaml(self.config_path)
        config_obj = core.create_and_validate_config(parsed_yaml)

        self.assertEqual(config_obj.app_config_.training_data_file, "bike-sharing-dataset.csv")
        self.assertEqual(config_obj.app_config_.pipeline_save_file, "bike_Sharing_model_output_v")
        self.assertEqual(config_obj.model_config_.target, "cnt")
        self.assertEqual(config_obj.model_config_.features,
                         ["dteday", "season", "hr", "holiday", "weekday", "workingday",
                          "weathersit", "temp", "atemp", "hum", "windspeed", "casual", "registered"])
        self.assertEqual(config_obj.model_config_.test_size, 0.20)
        self.assertEqual(config_obj.model_config_.random_state, 42)
        self.assertEqual(config_obj.model_config_.n_estimators, 150)
        self.assertEqual(config_obj.model_config_.max_depth, 5)
        self.assertEqual(config_obj.model_config_.max_features, 3)

    if __name__ == "__main__":
        unittest.main()
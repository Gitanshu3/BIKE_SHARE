# Prediction script
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
import pandas as pd
import numpy as np


from bike_sharing_model import __version__ as _version
from bike_sharing_model.config.core import config
from pipeline import bike_sharing_model_pipeline
from bike_sharing_model.processing.data_manager import load_pipeline
from bike_sharing_model.processing.data_manager import pre_pipeline_preparation
from bike_sharing_model.processing.validation import validate_Inputs


pipeline_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
bike_sharing_model_pipe= load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model """

    validated_data, errors = validate_Inputs(input_df=pd.DataFrame(input_data))

    validated_data=validated_data.reindex(columns=config.model_config_.features)

    predictions = bike_sharing_model_pipe.predict(validated_data)
    results = {"predictions": predictions, "version": _version}
    # results = {"predictions": predictions,"version": _version, "errors": errors}
    print(results)
    if not errors:

        predictions = bike_sharing_model_pipe.predict(validated_data)
        results = {"predictions": predictions, "version": _version}
        # results = {"predictions": predictions, "version": _version, "errors": errors}
        print(results)

    return results


if __name__ == "__main__":
     data_in = {'dteday': '2011-07-13', 'season': 'fall', 'hr': '4am', 'holiday': 'No', 'weekday': 'Wed',
                          'workingday': 'Yes', 'weathersit': 'Clear', 'temp': 26.78, 'atemp': 28.9988, 'hum': 58.0,
                          'windspeed': 16.9979, 'casual': 0, 'registered': 5, 'year':2012, 'month':4}
     make_prediction(input_data=data_in)
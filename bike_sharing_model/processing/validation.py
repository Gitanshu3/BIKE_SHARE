# Data validation functions
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np
from pydantic import ValidationError, BaseModel

from bike_sharing_model import config
from bike_sharing_model.processing.data_manager import pre_pipeline_preparation


class InputDataSchema(BaseModel):
    dteday: Optional[str]
    season: Optional[str]
    hr: Optional[str]
    holiday: Optional[str]
    weekday: Optional[str]
    workingday: Optional[str]
    weathersit: Optional[str]
    temp: Optional[float]
    atemp: Optional[float]
    hum: Optional[float]
    windspeed: Optional[float]
    casual: Optional[int]
    registered: Optional[int]


class MultipleDataInputs(BaseModel):
    inputs: List[InputDataSchema]


def validate_Inputs(*, input_df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    pre_processed_data = pre_pipeline_preparation(df=input_df)
    validated_data = pre_processed_data[config.model_config_.features].copy()
    errors = None

    try:
        MultipleDataInputs(inputs=validated_data.replace({np.nan: None}).to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors

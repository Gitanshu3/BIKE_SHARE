# Data loading and preprocessing functions
import sys
from pathlib import Path
import os

from numpy import datetime64
from sklearn.pipeline import Pipeline

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import joblib
import pandas as pd
import typing as t

from bike_sharing_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR
from bike_sharing_model import __version__ as _version
from bike_sharing_model import config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    transformed = pre_pipeline_preparation(df=dataframe)
    return transformed


#use for inference if csv is available
def load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def pre_pipeline_preparation(*, df: pd.DataFrame) -> pd.DataFrame:
    # if 'dteday' in df.columns:
    #     df['dteday'] = df['dteday'].astype('datetime64[ns]').astype(int)
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday']).astype(int)
        df['year'] = pd.to_datetime(df['dteday']).dt.year
        df['month'] = pd.to_datetime(df['dteday']).dt.month
    return df


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    # remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
    print("Model/pipeline trained successfully!")


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    do_not_delete = files_to_keep + ["__init__.py", ".gitignore"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            os.chmod(model_file, 0o777)
            model_file.unlink()


def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model

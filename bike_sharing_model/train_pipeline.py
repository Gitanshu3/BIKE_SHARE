# Training script
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.model_selection import train_test_split

from bike_sharing_model.pipeline import bike_sharing_model_pipeline

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from bike_sharing_model.processing.data_manager import load_dataset, save_pipeline
from bike_sharing_model.config.core import config

def run_training() -> None:

    data = load_dataset(file_name = config.app_config_.training_data_file)
    x_train, x_test, y_train, y_test = train_test_split(
        data[config.model_config_.features],
        data[config.model_config_.target],
        test_size=config.model_config_.test_size,
        random_state=config.model_config_.random_state
    )

    bike_sharing_model_pipeline.fit(x_train, y_train)

    save_pipeline(pipeline_to_persist= bike_sharing_model_pipeline)


if __name__ == "__main__":
    run_training()
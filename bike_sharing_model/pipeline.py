# Model pipeline definition
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from bike_sharing_model.processing.features import Mapper, WeekdayImputer, WeathersitImputer, OutlierHandler, WeekdayOneHotEncoder

numerical_features = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
outlier_handler = OutlierHandler(variables=numerical_features)

mappings_dict = {
    "season": {1: 0, 2: 1, 3: 2, 4: 3},
    "holiday": {0: 0, 1: 1},
    "workingday": {0: 0, 1: 1},
    "weathersit": {1: 0, 2: 1, 3: 2, 4: 3},
    "year": {2011: 0, 2012: 1},
    "month": {i: i-1 for i in range(1, 13)},
    "hr": {str(i): i for i in range(24)}  # Map hour strings to integers
}

mapper = Mapper(
    variables=["season", "holiday", "workingday", "weathersit", "year", "month", "hr"],
    mappings=mappings_dict
)

bike_sharing_model_pipeline = Pipeline([
    ('weekday_imputer', WeekdayImputer()),
    ('weathersit_imputer', WeathersitImputer()),
    ('mapper', mapper),
    ('outlier_handler',  outlier_handler),
    ('onehot_encoder', WeekdayOneHotEncoder()),
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])
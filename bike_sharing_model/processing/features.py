# Feature engineering functions
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class WeekdayImputer(BaseEstimator, TransformerMixin):

    def __init__(self, date_col='dteday', weekday_col='weekday'):
        self.date_col = date_col
        self.weekday_col = weekday_col

    def fit(self, X, y=None):
        return self  # No learned parameters, just return self

    def transform(self, X):
        X = X.copy()
        missing_idx = X[self.weekday_col].isna()
        if missing_idx.any():  # If there are NaN values
            X.loc[missing_idx, self.weekday_col] = (
                pd.to_datetime(X.loc[missing_idx, self.date_col])
                .dt.day_name().str[:3]  # Convert to short format
            )
        return X


class WeathersitImputer(BaseEstimator, TransformerMixin):

    def __init__(self, weathersit_col='weathersit'):
        self.weathersit_col = weathersit_col

    def fit(self, X, y=None):
        self.most_frequent = X[self.weathersit_col].mode()[0]  # Find most frequent value
        return self

    def transform(self, X):
        X = X.copy()
        X[self.weathersit_col].fillna(self.most_frequent, inplace=True)
        return X

class Mapper(BaseEstimator, TransformerMixin):

    def __init__(self, variables: list, mappings: dict):
        if not isinstance(variables, list):
            raise ValueError("variables should be a list")
        self.variables = variables
        self.mappings = mappings

    def fit(self, X, y=None):
        return self  # Nothing to learn

    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            if feature in X.columns:
                X[feature] = X[feature].map(self.mappings[feature])
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):

    def __init__(self, variables=None): # Added variables argument
        if variables is None:
            self.variables = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered', 'cnt'] # Default numerical features if None is provided
        else:
            self.variables = variables

    def fit(self, X, y=None):
        # Only calculate bounds for specified numerical features
        # Select only the columns that are present in the DataFrame
        valid_cols = [col for col in self.variables if col in X.columns]
        self.lower_bounds = X[valid_cols].quantile(0.05)
        self.upper_bounds = X[valid_cols].quantile(0.95)
        return self

    def transform(self, X):
        X = X.copy()
        # Only clip values for specified numerical features
        for col in self.variables:
            if col in X.columns: # Check if the column exists in X
                X[col] = np.clip(X[col], self.lower_bounds[col], self.upper_bounds[col])
        return X


class WeekdayOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, prefix='weekday'):
        self.prefix = prefix

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Get one-hot encoded columns
        encoded_df = pd.get_dummies(X['weekday'], prefix=self.prefix, drop_first=False)

        # Define the full set of expected columns
        expected_cols = ['weekday_Fri', 'weekday_Mon', 'weekday_Sat', 'weekday_Sun',
                         'weekday_Thu', 'weekday_Tue', 'weekday_Wed']

        # Ensure all expected columns exist in the DataFrame
        for col in expected_cols:
            if col not in encoded_df.columns:
                encoded_df[col] = 0  # Add missing columns with 0 values

        # Reorder columns to match training order
        encoded_df = encoded_df[expected_cols]

        # Drop the original weekday column
        X = X.drop('weekday', axis=1)

        # Concatenate the encoded columns with the original DataFrame
        X = pd.concat([X, encoded_df], axis=1)
        return X
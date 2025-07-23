import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import yaml
import logging

logger = logging.getLogger('model_prediction')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_prediction.log')
logger.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)["model_prediction"]
        return params
    except FileNotFoundError:
        logger.error(f"Parameter file not found: {params_path}")
        raise
    except KeyError:
        logger.error("Missing 'model_prediction' key in params.yaml")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise

def read_csv(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    try:
        for df in [train, test]:
            if not isinstance(df, pd.DataFrame):
                raise ValueError("Input to read_csv must be DataFrames.")
            df.replace(['unknown', 'badval', 'error'], np.nan, inplace=True)
        return train, test
    except Exception as e:
        logger.error(f"Error cleaning data: {e}")
        raise

def split_data(train: pd.DataFrame) -> tuple:
    try:
        if 'efficiency' not in train.columns:
            raise KeyError("'efficiency' column not found in training data.")
        X = train.drop(columns=['efficiency'])
        y = train['efficiency']
        return X, y
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

def process(X: pd.DataFrame, y: pd.Series, params: dict) -> None:
    try:
        categorical_cols = ['string_id', 'error_code', 'installation_type']
        numerical_cols = [col for col in X.columns if col not in categorical_cols + ['id']]

        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, numerical_cols),
            ('cat', categorical_pipeline, categorical_cols)
        ])

        model_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=params['n_estimators'],
                random_state=params['random_state']
            ))
        ])

        model_pipeline.fit(X, y)
        pickle.dump(model_pipeline, open('model.pkl', 'wb'))
        logger.info('Pickle file saved successfully')
    except KeyError as e:
        logger.error(f"Missing parameter in config: {e}")
        raise
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

def main():
    try:
        params = load_params("params.yaml")

        train_df = pd.read_csv("data/imputations/train_missing_imputation.csv")
        test_df = pd.read_csv("data/imputations/test_missing_imputation.csv")

        train_df, test_df = read_csv(train_df, test_df)
        X, y = split_data(train_df)
        process(X, y, params)

        logger.info("Model training and saving completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
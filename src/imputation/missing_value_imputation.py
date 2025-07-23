import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
import os
import logging

logger = logging.getLogger('missing_value_imputation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('missing_value_imputation.log')
logger.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def read_csv(train:pd.DataFrame, test:pd.DataFrame)->pd.DataFrame:
    try:
        train.drop(columns=['temperature'], inplace=True)
        test.drop(columns=['temperature'], inplace=True)
        return train, test
    except Exception as e:
        logger.error(f"file is not found {e}")
        raise

def imputer(train:pd.DataFrame, test:pd.DataFrame, *cols:str)->pd.DataFrame:
    try:
        imputer = KNNImputer(n_neighbors=3)
        for col in cols:
            train[[col]] = imputer.fit_transform(train[[col]]) #humidity

        features_for_knn = ['irradiance', 'panel_age', 'soiling_ratio',
                            'voltage', 'current', 'module_temperature', 'cloud_coverage',
                            'wind_speed', 'pressure','maintenance_count','power']

        # Filter numeric data and drop non-numeric or irrelevant columns
        train_knn = train[features_for_knn].copy()
        test_knn = test[features_for_knn].copy()
        # Apply KNNImputer
        knn_imputer = KNNImputer(n_neighbors=3)
        train_imputed = knn_imputer.fit_transform(train_knn)
        test_imputed = knn_imputer.fit_transform(test_knn)
        
        train[features_for_knn] = pd.DataFrame(train_imputed, columns=features_for_knn)
        test[features_for_knn] = pd.DataFrame(test_imputed, columns=features_for_knn)
        
        return train, test
    except Exception as e:
        logger.error(f"Runtime error in Imputation{e}")
        raise
    

def process(train:pd.DataFrame, test:pd.DataFrame, *cols:str)->pd.DataFrame:
    # Replace the original values with imputed ones
    # Fill missing error_code with the most frequent value
    try:
        for col in cols:
            train[col] = train[col].fillna(train[col].mode()[0]) # error_code
            test[col] = test[col].fillna(test[col].mode()[0])
            # Fill missing installation_type similarly
            train[col] = train[col].fillna(train[col].mode()[0]) #installation_type
            test[col] = test[col].fillna(test[col].mode()[0])
        
        return train, test
    except Exception as e:
        logger.error(f"Run time error is occured{e}")
        raise

def save_data(data_path:str, train:pd.DataFrame, test:pd.DataFrame)->str:
    try:
        os.makedirs(data_path)
        train.to_csv(os.path.join(data_path, "train_missing_imputation.csv"))
        test.to_csv(os.path.join(data_path, "test_missing_imputation.csv"))
        logger.info(f"file saved successfully {data_path}")
    except Exception as e:
        logger.error(f"error is occured{e}")
        raise
    
    
def main():
    try:
        train = pd.read_csv("data/processed/train1.csv")
        test = pd.read_csv("data/processed/test1.csv")
        train, test = read_csv(train, test)
        train, test = imputer(train, test, 'humidity')
        train,test = process(train, test, 'error_code', 'installation_type')
        data_path = os.path.join("data",'imputations')
        save_data(data_path, train, test)
    except Exception as e:
        logger.error(f"File is not saved{e}")
        raise
    

if __name__ == "__main__":
    main()
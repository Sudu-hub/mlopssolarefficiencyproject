import numpy as np
import pandas as pd
import os
import logging

# Configure logging
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('features_engineering.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def read_csv(train:pd.DataFrame, test:pd.DataFrame)->pd.DataFrame:
    try:
        train['corrected_voltage'] = (train['voltage'] + 0.3 * (25 - train['module_temperature'])).clip(lower=0)
        test['corrected_voltage'] = (test['voltage'] + 0.3 * (25 - test['module_temperature'])).clip(lower=0)
        train = train[train['efficiency'] != 0.000000]
        train = train[train['corrected_voltage'] < 70]
        return train, test
    except Exception as e:
        logger.error(f'File is not found{e}')
        raise


def process(train:pd.DataFrame, test:pd.DataFrame, *cols:str)->pd.DataFrame:
    try:
        for col in cols:
            
            train['power'] = train[col] * train[col]
            test['power'] = test[col] * test[col]
            
        Q1 = train['power'].describe()['25%']
        Q3 = train['power'].describe()['75%']
        IQR = Q3 - Q1
            
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        train['power'] = train['power'].clip(lower=lower_bound, upper=upper_bound)
        return train, test
    except Exception as e:
        logger.error(f'Data is not formated{e}')
        raise

def save_data(data_path:str, train_data:str, test_data:str)->str:
    try:
        os.mkdir(data_path)
        train_data.to_csv(os.path.join(data_path, "train1.csv"))
        test_data.to_csv(os.path.join(data_path, "test1.csv"))
        logger.info(f"File saved successfully {data_path}")
    except Exception as e:
        logger.error(f'path is not identified{e}')
        raise
    
    
def main():
    try:
        train = pd.read_csv("data/raw/train.csv")
        test = pd.read_csv("data/raw/test.csv")
        train, test = read_csv(train, test)
        train, test = process(train,test, 'corrected_voltage','current')
        data_path = os.path.join("data","processed")
        save_data(data_path, train, test)
    except Exception as e:
        logger.error(f'Path not speciefied{e}')
        raise

if __name__ == "__main__":
    main()
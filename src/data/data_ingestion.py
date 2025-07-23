import numpy as np
import pandas as pd
import os
import logging

# Configure logging
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('data_ingestion.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Function to filter irradiance > 0
def read_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        train = train[train['irradiance'] > 0]
        test = test[test['irradiance'] > 0]
        return train, test
    except Exception as e:
        logger.error(f"Error filtering data: {e}")
        raise

# Function to convert specific columns to numeric
def process(train_df: pd.DataFrame, test_df: pd.DataFrame, *cols: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        for col in cols:
            train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error processing columns: {e}")
        raise

# Function to save data to CSV
def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)
        logger.info(f"Data saved successfully in {data_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

# Main function
def main():
    try:
        train_path = r"C:\Users\sudar\OneDrive\Documents\Courses\Hackathon\solar_panel_prediction\Master_dataset\train.csv"
        test_path = r"C:\Users\sudar\OneDrive\Documents\Courses\Hackathon\solar_panel_prediction\Master_dataset\test.csv"
        
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)

        train, test = read_data(train, test)
        train_df, test_df = process(train, test, 'humidity', 'wind_speed', 'pressure')

        data_path = os.path.join("data", "raw")
        save_data(data_path, train_df, test_df)

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise

if __name__ == "__main__":
    main()


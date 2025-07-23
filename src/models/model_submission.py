import numpy as np
import pandas as pd
import os
import pickle
import logging

logger = logging.getLogger('model_submission')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_submission.log')
logger.setLevel(logging.ERROR)

formater = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formater)
file_handler.setFormatter(formater)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Load test data and trained model
def read_data(test:pd.DataFrame, rf:str)->str:
    try:
        return test, rf
    except Exception as e:
        logger.error(f"Run time error{e}")
        raise

# Create submission DataFrame
def save_submit(data_path:str,rf:str, test:pd.DataFrame)->str:
    try:
        test_ids = test['id']
        X_test = test.drop(columns=['id'])
        test_preds = rf.predict(X_test)
        submission = pd.DataFrame({
            'id': test_ids,
            'efficiency': test_preds
        })
        os.makedirs(data_path, exist_ok=True)

        submission.to_csv(os.path.join(data_path, 'submission.csv'), index=False)
        logger.info('Submission file saved successfully {data_path}')
    except Exception as e:
        logger.error(f"file path is not uploaded{e}")
        raise
# Create folder and save CSV

def main():
    try:
        test = pd.read_csv("data/imputations/train_missing_imputation.csv")
        rf = pickle.load(open('model.pkl', 'rb'))
        test,rf = read_data(test, rf)
        data_path = os.path.join('data', 'submission')
        save_submit(data_path, rf, test)
        logger.info('saved files')
    except Exception as e:
        logger.error(f"data is not saved{e}")
        raise

if __name__ == "__main__":
    main()
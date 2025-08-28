import os
import sys
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime

root_path = "/home/dball/Formation/MLOps/Exams/examen-dvc/"
sys.path.append(root_path)

from tools.logger import setup_logger
logger = setup_logger(name="getRawData")

def getRawData(DestPath):
    logger.info("downloading raw data from bucket")
    try:
        response = requests.get("https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv")
        if response.status_code == 200:
            content = (response.content)
            with open(DestPath, "wb") as file:
                file.write(content)
        else:
            print(f"Error accessing the bucket:", response.status_code)
        
        df = pd.read_csv(DestPath)
        df.drop(columns=['date'], inplace=True)

        return df
    
    except Exception as e:
        logger.error("Error loading : %s", str(e), exc_info=True)
        raise

def splitData(df):
    logger.info("spliting data")
    try:
        X = df.drop(columns=['silica_concentrate'])
        y = df['silica_concentrate']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    
    except Exception as e:
        logger.error("Error spliting: %s", str(e), exc_info=True)
        raise

def saveData(X_train, X_test, y_train, y_test):
    logger.info("saving data")
    X_train.to_csv(os.path.join(root_path, "data/processed_data/X_train.csv"), index=False)
    X_test.to_csv(os.path.join(root_path, "data/processed_data/X_test.csv"), index=False)
    y_train.to_csv(os.path.join(root_path, "data/processed_data/y_train.csv"), index=False)
    y_test.to_csv(os.path.join(root_path, "data/processed_data/y_test.csv"), index=False)

def main():
    df = getRawData(os.path.join(root_path, "data/raw_data/raw.csv"))
    X_train, X_test, y_train, y_test = splitData(df)
    saveData(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()


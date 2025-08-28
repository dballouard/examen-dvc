import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

root_path = "/home/dball/Formation/MLOps/Exams/examen-dvc/"
sys.path.append(root_path)

from tools.logger import setup_logger
logger = setup_logger(name="setFeatures")

def scaling_features(X_train, X_test):
    logger.info("scaling features")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train_scaled, X_test_scaled

def saving_scaled_features(X_train_scaled, X_test_scaled):
    logger.info("saving scaled features")
    X_train_scaled.to_csv(os.path.join(root_path, "data/processed_data/X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(root_path, "data/processed_data/X_test_scaled.csv"), index=False)

def main():
    X_train = pd.read_csv(os.path.join(root_path, "data/processed_data/X_train.csv"))
    X_test = pd.read_csv(os.path.join(root_path, "data/processed_data/X_test.csv"))
    
    X_train_scaled, X_test_scaled = scaling_features(X_train, X_test)
    saving_scaled_features(X_train_scaled, X_test_scaled)

if __name__ == "__main__":
    main()

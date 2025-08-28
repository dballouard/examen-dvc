import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

root_path = "/home/dball/Formation/MLOps/Exams/examen-dvc/"
sys.path.append(root_path)

from tools.logger import setup_logger
logger = setup_logger(name="trainModel")

logger.info("Loading processed data for training")
X_train_scaled = pd.read_csv(root_path + "data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv(root_path + "data/processed_data/y_train.csv").values.ravel()

logger.info("Loading best parameters")
with open(os.path.join(root_path,"models/best_rf_params.pkl"), 'rb') as f:
    best_params = pickle.load(f)

rf = RandomForestRegressor(**best_params, random_state=42)
logger.info("Starting model training")
rf.fit(X_train_scaled, y_train)
logger.info("Model training completed")

logger.info("Saving trained model")
with open(os.path.join(root_path,"models/trained_rf_model.pkl"), 'wb') as f:
    pickle.dump(rf, f)



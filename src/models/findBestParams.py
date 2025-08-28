import numpy as np
import pandas as pd
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import pickle

root_path = "/home/dball/Formation/MLOps/Exams/examen-dvc/"
sys.path.append(root_path)

from tools.logger import setup_logger
logger = setup_logger(name="findBestParams")

logger.info("Loading processed data")
X_train_scaled = pd.read_csv(root_path + "data/processed_data/X_train_scaled.csv")
X_test_scaled = pd.read_csv(root_path + "data/processed_data/X_test_scaled.csv")
y_train = pd.read_csv(root_path + "data/processed_data/y_train.csv").values.ravel()
y_test = pd.read_csv(root_path + "data/processed_data/y_test.csv").values.ravel()

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['log2', 'sqrt'],
    'bootstrap': [True]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1,
    verbose=1
)
logger.info("Starting Grid Search")
grid_search.fit(X_train_scaled, y_train)
logger.info("Grid Search completed")

logger.info(f"Best R² score (CV) : {grid_search.best_score_:.4f}")
logger.info(f"Best params : {grid_search.best_params_}")
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
logger.info(f"R² on test : {r2:.4f}")

logger.info("Saving best params")
with open(os.path.join(root_path,"models/best_rf_params.pkl"), 'wb') as f:
    pickle.dump(grid_search.best_params_, f)

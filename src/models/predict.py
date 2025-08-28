import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle
import json
from datetime import datetime

root_path = "/home/dball/Formation/MLOps/Exams/examen-dvc/"
sys.path.append(root_path)

from tools.logger import setup_logger
logger = setup_logger(name="predict")

logger.info("Loading test dataset for prediction")
X_test_scaled = pd.read_csv(root_path + "data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv(root_path + "data/processed_data/y_test.csv").values.ravel()

logger.info("Loading trained model")
with open(os.path.join(root_path,"models/trained_rf_model.pkl"), 'rb') as f:
    rf = pickle.load(f)

logger.info("Starting prediction")
y_pred = rf.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
logger.info(f"On test dataset : R² : {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

logger.info("Saving prediction and results")
y_pred_df = pd.DataFrame(y_pred, columns=['Predicted_Silica_Concentrate'])
y_pred_df.to_csv(os.path.join(root_path,"data/y_pred.csv"), index=False)

metrics = {
    "model_name": "RandomForestRegressor",
    "date": datetime.now().strftime('%Y%m%d_%H%M%S'), 
    "metrics": {
        "r2": r2,
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }
}
with open(os.path.join(root_path,"metrics/scores.json"), 'w') as f:
    json.dump(metrics, f, indent=4)



"""
Hopsworks Model Serving predictor script.
Hopsworks loads this class inside the deployment container and calls
.predict() whenever a request comes in.
"""

import os
import joblib
import numpy as np


class Predict(object):
    def __init__(self):
        # Hopsworks mounts the registered model artifact at this path
        # inside the deployment container automatically
        model_dir = os.environ["MODEL_FILES_PATH"]
        model_file = [f for f in os.listdir(model_dir) if f.endswith(".pkl")][0]
        self.model = joblib.load(os.path.join(model_dir, model_file))

    def predict(self, inputs):
        # inputs comes in as a list of feature rows (list of lists)
        data = np.array(inputs)
        preds = self.model.predict(data)
        return preds.tolist()
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np


class ModelTester:

    def evaluate(self, model, X_test, y_test):

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        rmse = np.sqrt(
            mean_squared_error(y_test, predictions)
        )

        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
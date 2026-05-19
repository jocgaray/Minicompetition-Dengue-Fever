import pandas as pd


class TestLoader:

    def __init__(self, features_path: str):
        self.features_path = features_path

    def load(self):

        X_test = pd.read_csv(self.features_path)
        X_test=X_test.drop(columns=["week_start_date"])

        print("Test data loaded:")
        print(X_test.shape)

        return X_test
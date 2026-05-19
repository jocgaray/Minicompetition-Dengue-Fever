import pandas as pd

class DataLoader:

    def __init__(self, features_path: str, labels_path: str):
        self.features_path = features_path
        self.labels_path = labels_path

    def load(self):

        features = pd.read_csv(self.features_path)
        labels = pd.read_csv(self.labels_path)

        df = features.merge(
            labels,
            on=["city", "year", "weekofyear"]
        )

        print("Data loaded:")
        print(df.shape)

        return df
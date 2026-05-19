from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


class ModelTrainer:

    def split_data(self, df, target="total_cases"):

        X = df.drop(columns=[target, "date"])
        y = df[target]

        X = X.select_dtypes(include=["number"])

        return train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

    def train(self, X_train, y_train):

        model = RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        return model
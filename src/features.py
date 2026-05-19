class FeatureEngineer:

    def add_lag_features(self, df, columns, lags=[1, 2, 3]):

        for column in columns:
            for lag in lags:

                df[f"{column}_lag_{lag}"] = (
                    df.groupby("city")[column]
                    .shift(lag)
                )

        return df

    def add_rolling_features(self, df, columns, windows=[3, 5]):

        for column in columns:
            for window in windows:

                df[f"{column}_rolling_mean_{window}"] = (
                    df.groupby("city")[column]
                    .transform(
                        lambda x: x.rolling(window).mean()
                    )
                )

        return df
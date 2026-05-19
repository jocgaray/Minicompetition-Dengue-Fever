class DataCleaner:

    def create_datetime(self, df):

        df["date"] = pd.to_datetime(
            df["year"].astype(str) + "-01-01"
        ) + pd.to_timedelta(
            (df["weekofyear"] - 1) * 7,
            unit="D"
        )

        df = df.sort_values(["city", "date"])

        return df

    def remove_duplicates(self, df):
        return df.drop_duplicates()

    def fill_missing(self, df):

        numeric_cols = df.select_dtypes(include=["number"]).columns

        df[numeric_cols] = df[numeric_cols].fillna(
            df[numeric_cols].median()
        )

        return df
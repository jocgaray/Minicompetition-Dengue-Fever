class DataExplorer:

    def validate_data(self, df):

        print("\nMissing values:")
        print(df.isnull().sum().sort_values(ascending=False).head(20))

        print("\nDuplicate rows:")
        print(df.duplicated().sum())

        print("\nData types:")
        print(df.dtypes)

        return df

    def characterize_target(self, df, target="total_cases"):

        print("\nTarget statistics:")
        print(df[target].describe())

        return df
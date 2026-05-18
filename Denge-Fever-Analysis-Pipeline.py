from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

savePlotfile = True


@dataclass
class DengueAnalysisPipeline:
    features_path: str
    labels_path: str

    def load_data(self):

        features = pd.read_csv(self.features_path)
        labels = pd.read_csv(self.labels_path)

        self.df = features.merge(labels, on=["city", "year", "weekofyear"])

        print("Data loaded:")
        print(self.df.shape)

        return self

    def create_datetime(self):

        self.df["date"] = pd.to_datetime(
            self.df["year"].astype(str) + "-01-01"
        ) + pd.to_timedelta((self.df["weekofyear"] - 1) * 7, unit="D")

        self.df = self.df.sort_values(["city", "date"])

        return self

    def validate_data(self):

        print("\nMissing values:")
        print(self.df.isnull().sum().sort_values(ascending=False).head(20))

        print("\nDuplicate rows:")
        print(self.df.duplicated().sum())

        print("\nData types:")
        print(self.df.dtypes)

        return self

    def characterize_target(self):

        print("\nTarget statistics:")
        print(self.df["total_cases"].describe())

        plt.figure(figsize=(10, 5))

        sns.histplot(self.df["total_cases"], bins=50, kde=True)

        plt.title("Distribution of Dengue Cases")

        if savePlotfile:
            plt.savefig("plots/characterize_target.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return self

    def visualize_time_series(self):

        plt.figure(figsize=(15, 5))

        for city in self.df["city"].unique():
            subset = self.df[self.df["city"] == city]

            plt.plot(subset["date"], subset["total_cases"], label=city)

        plt.legend()
        plt.title("Dengue Cases Over Time")
        plt.xlabel("Date")
        plt.ylabel("Cases")

        if savePlotfile:
            plt.savefig("plots/visualize_time_series.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return self

    def analyze_seasonality(self):

        seasonal = (
            self.df.groupby(["city", "weekofyear"])["total_cases"].mean().reset_index()
        )

        plt.figure(figsize=(14, 5))

        for city in seasonal["city"].unique():
            subset = seasonal[seasonal["city"] == city]

            plt.plot(subset["weekofyear"], subset["total_cases"], label=city)

        plt.legend()

        plt.title("Seasonal Dengue Patterns")
        plt.xlabel("Week of Year")
        plt.ylabel("Average Cases")

        if savePlotfile:
            plt.savefig("plots/analyze_seasonality.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return self

    def correlation_analysis(self):

        numeric_df = self.df.select_dtypes(include=np.number)

        corr = numeric_df.corr()

        plt.figure(figsize=(12, 10))

        sns.heatmap(
            corr[["total_cases"]].sort_values(by="total_cases", ascending=False),
            annot=True,
            cmap="coolwarm",
        )

        plt.title("Correlation With Dengue Cases")

        if savePlotfile:
            plt.savefig("plots/correlation_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return self

    def lagged_correlation_analysis(self, target="total_cases", max_lag=12):

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        df = self.df.copy()

        # IMPORTANT: enforce correct time order
        df = df.sort_values(["city", "year", "weekofyear"])

        numeric_cols = df.select_dtypes(include=np.number).columns
        numeric_cols = [c for c in numeric_cols if c != target]

        results = []

        for col in numeric_cols:
            for lag in range(max_lag + 1):
                shifted = df.groupby("city")[col].shift(lag)

                valid = pd.concat([shifted, df[target]], axis=1).dropna()

                if len(valid) < 10:
                    corr = np.nan
                else:
                    corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])

                results.append({"feature": col, "lag": lag, "correlation": corr})

        res_df = pd.DataFrame(results)

        # -------------------------
        # PLOT (this was missing!)
        # -------------------------

        pivot = res_df.pivot(index="feature", columns="lag", values="correlation")

        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot, cmap="coolwarm", center=0)

        plt.title("Lagged Correlation with Dengue Cases")
        plt.xlabel("Lag (weeks)")
        plt.ylabel("Feature")

        if savePlotfile:
            plt.savefig(
                "plots/lagged_correlation_analysis.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
        else:
            plt.show()

        return self

    def lag_analysis(self, feature="reanalysis_specific_humidity_g_per_kg", lag=4):

        city_df = self.df[self.df["city"] == "sj"].copy()

        city_df[f"{feature}_lag"] = city_df[feature].shift(lag)

        plt.figure(figsize=(8, 6))

        sns.scatterplot(data=city_df, x=f"{feature}_lag", y="total_cases")

        plt.title(f"{lag}-Week Lag Relationship")

        if savePlotfile:
            plt.savefig("plots/lag_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return self

    def rolling_analysis(self):

        city_df = self.df[self.df["city"] == "sj"].copy()

        city_df["temp_roll_4"] = city_df["station_avg_temp_c"].rolling(4).mean()

        plt.figure(figsize=(14, 5))

        plt.plot(city_df["date"], city_df["temp_roll_4"], label="Rolling Temp")

        plt.plot(city_df["date"], city_df["total_cases"], label="Cases")

        plt.legend()

        plt.title("Rolling Temperature vs Cases")

        if savePlotfile:
            plt.savefig("plots/rolling_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        return self


pipeline = (
    DengueAnalysisPipeline(
        features_path="Data/dengue_features_train.csv",
        labels_path="Data/dengue_labels_train.csv",
    )
    .load_data()
    .create_datetime()
    .validate_data()
    .characterize_target()
    .visualize_time_series()
    .analyze_seasonality()
    .correlation_analysis()
    .lagged_correlation_analysis()
    .lag_analysis()
    .rolling_analysis()
)

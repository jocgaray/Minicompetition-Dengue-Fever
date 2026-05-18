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

    def data_quality_check(self):

        import numpy as np
        import pandas as pd

        df = self.df.copy()

        report = {}

        # -----------------------------
        # 1. Missing values
        # -----------------------------
        report["missing"] = df.isna().sum().sort_values(ascending=False)

        # -----------------------------
        # 2. Detect non-numeric junk in numeric columns
        # -----------------------------
        numeric_cols = df.select_dtypes(include=np.number).columns

        bad_cols = []

        for col in numeric_cols:
            # try coercion check (safe way to detect hidden strings)
            coerced = pd.to_numeric(df[col], errors="coerce")

            if coerced.isna().sum() > df[col].isna().sum():
                bad_cols.append(col)

        report["potential_string_issues"] = bad_cols

        # -----------------------------
        # 3. Object columns overview
        # -----------------------------
        report["object_columns"] = df.select_dtypes(include="object").columns.tolist()

        self.quality_report = report

        return self

    def histogram_range_heatmap(self, bins=50):

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        df = self.df.copy()
        numeric_df = df.select_dtypes(include=np.number)

        heatmap_data = []

        for col in numeric_df.columns:
            counts, bin_edges = np.histogram(numeric_df[col].dropna(), bins=bins)

            heatmap_data.append(counts)

        heatmap_df = pd.DataFrame(heatmap_data, index=numeric_df.columns)

        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_df, cmap="magma")

        plt.title("Histogram Distribution Heatmap")
        plt.xlabel("Bins")
        plt.ylabel("Features")

        if savePlotfile:
            plt.savefig(
                "plots/histogram_range_heatmap.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
        else:
            plt.show()

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

    def analyze_seasonality(
        self, save_plot=False, save_path="plots/analyze_seasonality.png"
    ):

        import os

        import matplotlib.pyplot as plt

        df = self.df.copy()

        # -----------------------------
        # Aggregate seasonality
        # -----------------------------
        seasonal = (
            df.groupby(["city", "weekofyear"])["total_cases"].mean().reset_index()
        )

        seasonal = seasonal.sort_values(["city", "weekofyear"])

        # split cities
        sj = seasonal[seasonal["city"] == "sj"]
        iq = seasonal[seasonal["city"] == "iq"]

        # -----------------------------
        # SCALE IQ so it's visible
        # -----------------------------
        iq_scaled = iq.copy()
        iq_scaled["total_cases"] = (
            iq_scaled["total_cases"] / iq_scaled["total_cases"].max()
        ) * sj["total_cases"].max()

        # -----------------------------
        # PLOT
        # -----------------------------
        fig, ax1 = plt.subplots(figsize=(14, 5))

        # Axis 1: SJ + IQ (scaled)
        ax1.plot(sj["weekofyear"], sj["total_cases"], label="SJ cases", color="tab:red")
        ax1.plot(
            iq_scaled["weekofyear"],
            iq_scaled["total_cases"],
            label="IQ (scaled)",
            color="tab:orange",
        )

        ax1.set_xlabel("Week of Year")
        ax1.set_ylabel("Cases (scaled comparison)")
        ax1.legend(loc="upper left")

        # -----------------------------
        # Axis 2 (optional: smooth trend)
        # -----------------------------
        ax2 = ax1.twinx()

        ax2.plot(
            sj["weekofyear"],
            sj["total_cases"].rolling(4).mean(),
            linestyle="dashed",
            color="darkred",
            alpha=0.6,
            label="SJ trend",
        )

        ax2.set_ylabel("Smoothed Trend (SJ)")

        plt.title("Seasonal Dengue Pattern (SJ vs IQ + Trend)")

        # -----------------------------
        # SAVE
        # -----------------------------
        if savePlotfile:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()

        else:
            plt.show()
            plt.close()

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

        fig, ax1 = plt.subplots(figsize=(14, 5))

        ax1.plot(city_df["date"], city_df["total_cases"], color="tab:red")
        ax1.set_ylabel("Cases", color="tab:red")

        ax2 = ax1.twinx()
        ax2.plot(city_df["date"], city_df["temp_roll_4"], color="tab:blue")
        ax2.set_ylabel("Rolling Temp (°C)", color="tab:blue")

        plt.title("Cases vs Rolling Temperature")

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
    .histogram_range_heatmap()
    # .data_quality_check()
    # .characterize_target()
    # .visualize_time_series()
    # .analyze_seasonality()
    # .correlation_analysis()
    # .lagged_correlation_analysis()
    # .lag_analysis()
    # .rolling_analysis()*/
)


@dataclass
class VisualisationPipeline:
    def __init__(self, df, city=None):
        self.city = city
        self.df = df.copy()
        self.results = {}

    def savePlot(self, plt, plotname):
        save_path = f"plots_{self.city}/{plotname}.png"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    def histogram_range_heatmap(self, bins=50):

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        df = self.df
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
            self.savePlot(plt, "histogram_range_heatmap")

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
            self.savePlot(plt, "characterize_target")
            plt.close()
        else:
            plt.show()

        return self

    def visualize_time_series(self, save_plot=False):

        df = self.df

        plt.figure(figsize=(15, 5))

        plt.plot(df["date"], df["total_cases"], label="total_cases")

        plt.title(
            f"Dengue Cases Over Time ({df['city'].iloc[0] if 'city' in df.columns else 'city'})"
        )
        plt.xlabel("Date")
        plt.ylabel("Cases")
        plt.legend()

        if savePlotfile:
            self.savePlot(plt, "visualize_time_series")
            plt.close()
        else:
            plt.show()
            plt.close()

        return self

    def analyze_seasonality(self, save_plot=False):

        df = self.df.copy()

        if "city" in df.columns:
            city_name = df["city"].iloc[0]
        else:
            city_name = "unknown"

        # -----------------------------
        # Aggregate seasonality
        # -----------------------------
        seasonal = (
            df.groupby("weekofyear")["total_cases"]
            .mean()
            .reset_index()
            .sort_values("weekofyear")
        )

        # -----------------------------
        # Rolling trend
        # -----------------------------
        seasonal["trend"] = seasonal["total_cases"].rolling(4, min_periods=1).mean()

        # -----------------------------
        # PLOT
        # -----------------------------
        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(
            seasonal["weekofyear"],
            seasonal["total_cases"],
            label="Mean cases",
            color="tab:blue",
        )

        ax.plot(
            seasonal["weekofyear"],
            seasonal["trend"],
            linestyle="dashed",
            label="4-week trend",
            color="black",
            alpha=0.7,
        )

        ax.set_xlabel("Week of Year")
        ax.set_ylabel("Cases")
        ax.set_title(f"Seasonality Pattern ({city_name})")
        ax.legend()

        # -----------------------------
        # SAVE
        # -----------------------------
        #

        if savePlotfile:
            self.savePlot(plt, "analyze_seasonality")
            plt.close()
        else:
            plt.show()
            plt.close()

        return self

    def feature_correlation_analysis(self):

        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns

        df = self.df.copy()

        numeric_df = df.select_dtypes(include=np.number)

        # REMOVE target
        if "total_cases" in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=["total_cases"])

        corr = numeric_df.corr()

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0)

        plt.title("Feature Correlation Matrix (No Target Leakage)")

        if savePlotfile:
            self.savePlot(plt, "feature_correlation_analysis")
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
            self.savePlot(plt, "correlation_analysis")
            plt.close()
        else:
            plt.show()

        return self

    def lagged_correlation_analysis(self, target="total_cases", max_lag=12):

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        df = self.df

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
            self.savePlot(plt, "lagged_correlation_analysis.png")
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
            self.savePlot(plt, "lag_analysis.png")
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
            self.savePlot(plt, "rolling_analysis.png")
            plt.close()
        else:
            plt.show()

        return self

import matplotlib.pyplot as plt
import seaborn as sns


class Visualizer:

    def plot_time_series(self, df, column="total_cases"):

        plt.figure(figsize=(14, 6))

        sns.lineplot(
            data=df,
            x="date",
            y=column,
            hue="city"
        )

        plt.title(f"{column} over time")
        plt.show()

    def correlation_heatmap(self, df):

        corr = df.corr(numeric_only=True)

        plt.figure(figsize=(14, 10))

        sns.heatmap(corr, cmap="coolwarm")

        plt.title("Correlation Matrix")
        plt.show()
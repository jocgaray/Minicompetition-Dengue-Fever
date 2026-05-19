class DenguePipeline:

    def __init__(self, features_path, labels_path):

        self.loader = DataLoader(features_path, labels_path)
        self.explorer = DataExplorer()
        self.cleaner = DataCleaner()
        self.engineer = FeatureEngineer()
        self.visualizer = Visualizer()
        self.trainer = ModelTrainer()
        self.tester = ModelTester()

    def run(self):

        df = self.loader.load()

        df = self.cleaner.create_datetime(df)
        df = self.cleaner.remove_duplicates(df)
        df = self.cleaner.fill_missing(df)

        self.explorer.validate_data(df)
        self.explorer.characterize_target(df)

        df = self.engineer.add_lag_features(
            df,
            columns=["reanalysis_air_temp_k"]
        )

        df = self.engineer.add_rolling_features(
            df,
            columns=["reanalysis_air_temp_k"]
        )

        self.visualizer.plot_time_series(df)

        X_train, X_test, y_train, y_test = (
            self.trainer.split_data(df)
        )

        model = self.trainer.train(X_train, y_train)

        self.tester.evaluate(labels_path
            model,
            X_test,
            y_test
        )
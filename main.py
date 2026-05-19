from src.pipeline import DenguePipeline


pipeline = DenguePipeline(
    features_path="data/raw/dengue_features_train.csv",
    labels_path="data/raw/dengue_labels_train.csv"
)

pipeline.run()

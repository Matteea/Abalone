import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from progetto.data_manager import DataManager
from progetto.deep_learning import DeepLearningModel
from progetto.machine_learning import MachineLearningModels

import pandas as pd


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)

    data_manager = DataManager()
    data_manager.prepare_data()

    ml_models = MachineLearningModels(
        preprocessor=data_manager.preprocessor,
        X_train=data_manager.X_train,
        X_test=data_manager.X_test,
        y_train=data_manager.y_train,
        y_test=data_manager.y_test
    )

    ml_models.train_linear_regression()
    ml_models.train_xgboost()

    dl_model = DeepLearningModel(
        preprocessor=data_manager.preprocessor,
        X_train=data_manager.X_train,
        X_test=data_manager.X_test,
        y_train=data_manager.y_train,
        y_test=data_manager.y_test
    )

    dl_model.run()


if __name__ == "__main__":
    main()
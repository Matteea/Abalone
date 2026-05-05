
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import metrics

from xgboost import XGBRegressor

class MachineLearningModels:
    def __init__(self, preprocessor, X_train, X_test, y_train, y_test):
        self.preprocessor = preprocessor

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    #valutazione delle metriche del modello
    def evaluate_model(self, model_name, y_pred):
        mae = metrics.mean_absolute_error(self.y_test, y_pred)
        mse = metrics.mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(self.y_test, y_pred)

        print(f"\n{model_name}:")
        print("MAE:", mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R2:", r2)

    #stampa delle features utilizzate dal modello
    def print_feature_names(self, model):
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        print("\nFeature usate dal modello:")
        print(feature_names)

    #training della linear regression
    def train_linear_regression(self):
        model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("regressor", LinearRegression())
            ]
        )

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        self.evaluate_model("Linear Regression", y_pred)
        self.print_feature_names(model)

    #training del xgboost
    def train_xgboost(self):
        model = Pipeline(
            steps=[
                ("preprocessor", self.preprocessor),
                ("regressor", XGBRegressor(
                    objective="reg:squarederror",
                    random_state=12
                ))
            ]
        )

        model.fit(self.X_train, self.y_train)

        y_pred = model.predict(self.X_test)

        self.evaluate_model("XGBoost", y_pred)
        self.print_feature_names(model)
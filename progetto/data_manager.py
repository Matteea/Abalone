
from ucimlrepo import fetch_ucirepo
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


class DataManager:
    def __init__(self, test_size=0.2, random_state=12):
        self.test_size = test_size
        self.random_state = random_state

        self.df = None
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.preprocessor = None

    #caricamento del dataset
    def load_data(self):
        abalone = fetch_ucirepo(id=1)

        self.df = pd.DataFrame(abalone.data.features)
        rings = abalone.data.targets.squeeze()

        self.df["Rings"] = rings

        print("Shape iniziale:", self.df.shape)

    #pulizia del dataset
    def clean_data(self):
        df = self.df.copy()

        df = df[df["Height"] > 0].copy()

        partial_weights_sum = (
            df["Shucked_weight"] +
            df["Viscera_weight"] +
            df["Shell_weight"]
        )

        df = df[df["Whole_weight"] > partial_weights_sum].copy()

        weight_cols = [
            "Whole_weight",
            "Shucked_weight",
            "Viscera_weight",
            "Shell_weight"
        ]

        for col in weight_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            df = df[
                (df[col] >= lower_bound) &
                (df[col] <= upper_bound)
            ].copy()

        self.df = df

        print("Shape dopo pulizia:", self.df.shape)

    #ingegnerizzazione delle feature
    def feature_engineering(self):
        df = self.df.copy()

        df["Age"] = df["Rings"] + 1.5

        df["Volume"] = df["Length"] * df["Diameter"] * df["Height"]

        df["Shell_Ratio"] = df["Shell_weight"] / df["Whole_weight"]
        df["Shucked_Ratio"] = df["Shucked_weight"] / df["Whole_weight"]
        df["Viscera_Ratio"] = df["Viscera_weight"] / df["Whole_weight"]

        self.X = df.drop(columns=["Rings", "Age"])
        self.y = df["Age"]

        self.df = df

        print(self.X.head())
        print(self.y.head())

    #preparazione del preprocessor
    def build_preprocessor(self):
        categorical_features = ["Sex"]
        numeric_features = self.X.drop(columns=categorical_features).columns.tolist()

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
            ],
            remainder="drop"
        )

    #splitting dei dati
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X,
            self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )

    #preparazione dei dati
    def prepare_data(self):
        self.load_data()
        self.clean_data()
        self.feature_engineering()
        self.build_preprocessor()
        self.split_data()
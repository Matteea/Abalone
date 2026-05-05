
import numpy as np
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

class DeepLearningModel:
    def __init__(self, preprocessor, X_train, X_test, y_train, y_test):
        self.preprocessor = preprocessor

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.X_train_processed = None
        self.X_test_processed = None
        self.model = None

    #preprocess dei dati
    def preprocess_data(self):
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)

    #costruzione della rete neurale, 3 hidden layer da 64, 32 e 16 nodi
    def build_model(self):
        self.model = Sequential([
            Input(shape=(self.X_train_processed.shape[1],)),

            Dense(64, activation="relu"),
            Dropout(0.2),

            Dense(32, activation="relu"),
            Dropout(0.2),

            Dense(16, activation="relu"),

            Dense(1)
        ])

        self.model.compile(
            optimizer="adam",
            loss="mse",
            metrics=["mae"]
        )

    #allenamento del modello fino al raggiungimento di uno specifico valore di val_loss
    def train_model(self):
        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        )

        self.model.fit(
            self.X_train_processed,
            self.y_train,
            validation_split=0.2,
            epochs=300,
            batch_size=32,
            callbacks=[early_stop],
            verbose=1
        )

    #valutazione delle metriche del modello
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test_processed).flatten()

        mae = metrics.mean_absolute_error(self.y_test, y_pred)
        mse = metrics.mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = metrics.r2_score(self.y_test, y_pred)

        print("\nDeep Learning:")
        print("MAE:", mae)
        print("MSE:", mse)
        print("RMSE:", rmse)
        print("R2:", r2)

    #esecuzione completa della classe
    def run(self):
        self.preprocess_data()
        self.build_model()
        self.train_model()
        self.evaluate_model()
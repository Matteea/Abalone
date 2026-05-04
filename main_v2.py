from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from xgboost import XGBRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. Caricamento dati
# =========================

abalone = fetch_ucirepo(id=1)

df = pd.DataFrame(abalone.data.features)
rings = abalone.data.targets.squeeze()

df["Rings"] = rings

print("Shape iniziale:", df.shape)

# =========================
# 2. Pulizia dei dati
# =========================

# Rimuove record con Height = 0
df = df[df["Height"] > 0].copy()

# Controllo coerenza pesi
partial_weights_sum = (
    df["Shucked_weight"] +
    df["Viscera_weight"] +
    df["Shell_weight"]
)

df = df[df["Whole_weight"] > partial_weights_sum].copy()

# Gestione outlier con IQR sui pesi
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

print("Shape dopo pulizia:", df.shape)

# =========================
# 3. Feature engineering
# =========================

# Trasformazione target
df["Age"] = df["Rings"] + 1.5

# Feature geometriche
df["Volume"] = df["Length"] * df["Diameter"] * df["Height"]

# Rapporti di peso
df["Shell_Ratio"] = df["Shell_weight"] / df["Whole_weight"]
df["Shucked_Ratio"] = df["Shucked_weight"] / df["Whole_weight"]
df["Viscera_Ratio"] = df["Viscera_weight"] / df["Whole_weight"]

# Rimozione target originale dalle feature
#X = df.drop(columns=["Rings", "Age", "Shell_weight", "Shucked_weight", "Viscera_weight", "Whole_weight", "Length", "Diameter", "Height"])
X = df.drop(columns=["Rings", "Age"])
y = df["Age"]

pd.set_option('display.max_columns', None)   # mostra tutte le colonne
pd.set_option('display.max_rows', None)      # (opzionale) mostra tutte le righe
pd.set_option('display.width', None)         # evita il wrapping
pd.set_option('display.max_colwidth', None)  # mostra tutto il contenuto
print(X.head())
print(y.head())

# =========================
# 4. Encoding + Scaling
# =========================

categorical_features = ["Sex"]

numeric_features = X.drop(columns=categorical_features).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features)
    ],
    remainder="drop"
)

# =========================
# 5. Train-test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=12
)

# =========================
# 6. Modello
# =========================

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# =========================
# 7. Metriche
# =========================

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("Linear Regression:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

# =========================
# 8. Feature finali dopo preprocessing
# =========================

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

print("\nFeature usate dal modello:")
print(feature_names)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor())
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# =========================
# 7. Metriche
# =========================

mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("XGBoost:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)

# =========================
# 8. Feature finali dopo preprocessing
# =========================

feature_names = model.named_steps["preprocessor"].get_feature_names_out()

print("\nFeature usate dal modello:")
print(feature_names)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

model_dl = Sequential([
    Dense(64, activation="relu", input_shape=(X_train_processed.shape[1],)),
    Dropout(0.2),

    Dense(32, activation="relu"),
    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)   # output regressione
])

model_dl.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True
)

history = model_dl.fit(
    X_train_processed,
    y_train,
    validation_split=0.2,
    epochs=300,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

y_pred_dl = model_dl.predict(X_test_processed).flatten()

mae = metrics.mean_absolute_error(y_test, y_pred_dl)
mse = metrics.mean_squared_error(y_test, y_pred_dl)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred_dl)

print("Deep Learning:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)
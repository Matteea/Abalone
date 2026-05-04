from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# fetch dataset
abalone = fetch_ucirepo(id=1)
df = pd.DataFrame(abalone.data.features)
y = abalone.data.targets
print(df.head())
print(y)

print("valori nulli: ", df.isnull().sum())
print("valori duplicati: ", df.duplicated().sum())
print(df.info())



encoder = OneHotEncoder(sparse_output=False, drop="first")
encoded = encoder.fit_transform(df[["Sex"]])

encoded_df = pd.DataFrame(
    encoded,
    columns=encoder.get_feature_names_out(["Sex"])
)

df = pd.concat([df.drop("Sex", axis=1), encoded_df], axis=1)

print(df.head())

threshold = 0  # scegli tu (es. 0.3, 0.5, 0.7)

df_corr = df.copy()
df_corr["Rings"] = y.squeeze()

corr_matrix = df_corr.corr(numeric_only=True)

# creo una mask: nascondo valori sotto soglia
mask = np.abs(corr_matrix) < threshold

plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)
plt.title(f"Correlazioni |r| >= {threshold}")
plt.tight_layout()
plt.show()

X = df.copy()
#X = df.drop(columns=["Viscera_weight", "Shucked_weight", "Whole_weight", "Diameter", "Length"])
#df["Volume"] = df["Length"] * df["Diameter"] * df["Height"]
#df["Density"] = df["Whole_weight"] / df["Volume"]
#X = df.drop(columns=["Height", "Diameter", "Length", "Whole_weight"])


#numeric_columns = df_corr.select_dtypes(include=["float64", "int64"]).columns

for col in X.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_corr[col])
    plt.title(f"Boxplot di {col}")
    plt.tight_layout()
    plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)
test_data_prediction = model.predict(X_test_scaled)

mae = metrics.mean_absolute_error(y_test, test_data_prediction)
mse = metrics.mean_squared_error(y_test, test_data_prediction)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, test_data_prediction)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


# Abalone Age Prediction — Report delle Metriche di Valutazione

## Panoramica del Progetto

Questo progetto affronta un problema di **regressione supervisionata**: predire l'**età di un abalone** (mollusco marino) a partire da misurazioni fisiche, evitando il costoso processo di conteggio degli anelli al microscopio.

- **Dataset**: [Abalone UCI Repository](https://archive.ics.uci.edu/dataset/1/abalone) (id=1)
- **Target**: `Age = Rings + 1.5`
- **Split**: 80% training / 20% test (`random_state=12`)

---

## Pipeline dei Dati (`data_manager.py`)

Prima di valutare i modelli, i dati vengono preparati attraverso i seguenti passaggi:

| Step | Descrizione |
|------|-------------|
| **Caricamento** | Dataset scaricato via `ucimlrepo` |
| **Pulizia** | Rimozione righe con `Height == 0`, filtraggio righe con `Whole_weight <= somma pesi parziali`, rimozione outlier IQR (×1.5) sui 4 campi peso |
| **Feature Engineering** | Aggiunta di `Volume = Length × Diameter × Height`, `Shell_Ratio`, `Shucked_Ratio`, `Viscera_Ratio` |
| **Preprocessing** | `StandardScaler` su feature numeriche, `OneHotEncoder(drop="first")` su `Sex` |

---

## Metriche di Valutazione

Tutti e tre i modelli vengono valutati sullo stesso test set usando le seguenti metriche:

### MAE — Mean Absolute Error
$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

- **Interpretazione**: errore medio in anni tra il valore reale e quello predetto.
- **Valore ideale**: più basso è, meglio è.
- **Punto di forza**: robusta agli outlier, di facile interpretazione nella stessa unità del target (anni).

---

### MSE — Mean Squared Error
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

- **Interpretazione**: media degli errori al quadrato — penalizza maggiormente gli errori grandi.
- **Valore ideale**: più basso è, meglio è.
- **Punto di forza**: differenziabile, utile come funzione di loss (usata anche nella compilazione del modello Keras: `loss="mse"`).
- **Limite**: espressa in anni², meno intuitiva del MAE.

---

### RMSE — Root Mean Squared Error
$$RMSE = \sqrt{MSE}$$

- **Interpretazione**: radice quadrata del MSE, riporta l'errore nella stessa unità del target (anni).
- **Valore ideale**: più basso è, meglio è.
- **Punto di forza**: combina la sensibilità agli outlier del MSE con la leggibilità del MAE.

---

### R² — Coefficiente di Determinazione
$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

- **Interpretazione**: proporzione di varianza del target spiegata dal modello.
- **Range**: da −∞ a 1. Un valore di **1.0** indica predizioni perfette; **0.0** equivale a predire sempre la media; valori negativi indicano performance peggiori di un predittore costante.
- **Punto di forza**: metrica normalizzata, facilmente comparabile tra modelli diversi.

---

## Modelli Implementati

### 1. Linear Regression (`machine_learning.py`)

Modello lineare basato su `sklearn.linear_model.LinearRegression`, inserito in una `Pipeline` con il preprocessor.

- **Architettura**: regressione lineare ordinaria (OLS).
- **Vantaggi**: interpretabile, veloce, baseline solida.
- **Limitazioni**: assume relazione lineare tra feature e target.

---

### 2. Ridge Regression (`machine_learning.py`)
 
Regressione lineare con regolarizzazione L2 tramite `sklearn.linear_model.Ridge`.
 
- **Configurazione**: `alpha=1.0` (controlla la forza della regolarizzazione)
- **Vantaggi**: riduce l'overfitting penalizzando i coefficienti grandi, stabile in presenza di multicollinearità (utile qui dato che le feature peso sono correlate tra loro).
- **Limitazioni**: non azzera i coefficienti, non effettua feature selection.
---
 
### 3. Lasso Regression (`machine_learning.py`)
 
Regressione lineare con regolarizzazione L1 tramite `sklearn.linear_model.Lasso`.
 
- **Configurazione**: `alpha=0.1`
- **Vantaggi**: porta alcuni coefficienti esattamente a zero, producendo un modello sparso con feature selection implicita.
- **Limitazioni**: può essere instabile con feature altamente correlate; richiede tuning accurato di `alpha`.
---

### 4. XGBoost (`machine_learning.py`)

Gradient boosting su alberi decisionali con `XGBRegressor`.

- **Configurazione**:
  - `objective="reg:squarederror"` (minimizza MSE)
  - `random_state=12`
- **Vantaggi**: gestisce relazioni non lineari, robusto agli outlier residui, spesso superiore alla regressione lineare su dati tabulari.
- **Limitazioni**: più lento da addestrare, maggiore rischio di overfitting rispetto al modello lineare.

---

### 5. Deep Learning Neural Network (`deep_learning.py`)

Rete neurale feedforward costruita con `tensorflow.keras`.

| Layer | Nodi | Attivazione | Note |
|-------|------|-------------|------|
| Input | n_features | — | |
| Hidden 1 | 64 | ReLU | + Dropout 0.2 |
| Hidden 2 | 32 | ReLU | + Dropout 0.2 |
| Hidden 3 | 16 | ReLU | |
| Output | 1 | Lineare | regressione |

- **Ottimizzatore**: Adam
- **Loss**: MSE
- **Metrica di monitoring**: MAE
- **Training**:
  - Max 300 epoche, batch size 32
  - `EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)`
  - Validation split: 20% del training set

---

## Come Leggere i Risultati

Quando esegui `main.py`, l'output a console mostra per ogni modello:

```
<Nome Modello>:
MAE:  <valore>    ← errore medio in anni
MSE:  <valore>    ← errore quadratico medio in anni²
RMSE: <valore>    ← deviazione tipica degli errori in anni
R2:   <valore>    ← proporzione di varianza spiegata [0, 1]
```

### Risultati Comparativi
 
| Modello | MAE ⬇️ | MSE ⬇️ | RMSE ⬇️ | R² ⬆️ |
|---------|--------|--------|---------|-------|
| Linear Regression | 1.4641 | 4.0410 | 2.0102 | 0.5772 |
| Ridge | 1.4641 | 4.0427 | 2.0106 | 0.5770 |
| Lasso | 1.5537 | 4.4930 | 2.1196 | 0.5299 |
| XGBoost | 1.6247 | 4.8185 | 2.1951 | 0.4959 |
| Deep Learning (NN) | ~1.4212 | ~3.9307 | ~1.9826 | ~0.5888 |
 
---

### Guida alla comparazione

| Metrica | Modello migliore |
|---------|-----------------|
| MAE | quello con valore più basso |
| MSE | quello con valore più basso |
| RMSE | quello con valore più basso |
| R² | quello con valore più alto (vicino a 1) |

> **Nota**: per un dataset biologico come Abalone, un MAE di ~1.5 anni e un R² > 0.55 sono considerati risultati buoni, data la naturale variabilità biologica del target.

---

## Struttura del Progetto

```
progetto/
├── main.py               # Entry point: orchestrazione dell'intera pipeline
├── data_manager.py       # Caricamento, pulizia, feature engineering, split
├── machine_learning.py   # Linear Regression + Ridge + Lasso + XGBoost
└── deep_learning.py      # Rete neurale con Keras/TensorFlow
```

---

## Dipendenze Principali

```
pandas
scikit-learn
xgboost
tensorflow / keras
ucimlrepo
numpy
```

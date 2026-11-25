# Stock Price Prediction using LSTM (Exploratory Analysis)

## Overview
This notebook demonstrates an exploratory implementation of a Long Short-Term Memory (LSTM) neural network for **univariate** stock price forecasting. The model predicts the next day's 'Close' price based on a rolling 100-day historical window using historical market data.

> **⚠️ Disclaimer:** This project is for **educational and demonstrative purposes only**. Univariate LSTMs are not reliable for live trading, and this code should not be used as financial advice.

---

## Dataset & Preprocessing
- **Dataset Link:** [Stock Market Dataset by Jackson Crow (Kaggle)](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
- **Target File:** `CCOR.csv` (ETF data)
- **Target Variable:** Daily 'Close' price.
- **Preprocessing:** 
  - Dates parsed as datetime objects and set as the index.
  - Feature scaled to a `[0, 1]` range using `MinMaxScaler` to aid neural network convergence.
- **Split:** 65% Training, 35% Testing.

---

## Model Architecture
Built using TensorFlow/Keras with the following configuration:
- **Input Shape:** `(100, 1)` → 100 time steps, 1 feature.
- **LSTM Layer 1:** 128 units, `return_sequences=True`.
- **LSTM Layer 2:** 64 units.
- **Dense Layer:** 32 units, `relu` activation.
- **Output Layer:** 1 unit, `sigmoid` activation *(Note: Constrains output to the 0–1 scaled range, which can cause vanishing gradients and limit prediction of boundary values).*
- **Optimizer:** Adam (`learning_rate=0.001`).
- **Loss Function:** Mean Squared Error (MSE).

---

## Evaluation Metrics
- **Train RMSE:** ~0.159 
- **Test RMSE:** ~0.490 

*Interpretation:* The Test RMSE represents an average error of roughly ~1.7% relative to the ~$28 stock price. However, the noticeable gap between Train and Test RMSE indicates **mild overfitting**, where the model memorizes training noise rather than generalizing perfectly.

---

## Critical Limitations & Caveats
Anyone reviewing or extending this code must be aware of the following methodological constraints:

1. **Test Set Context Gap:** The data is split into train/test *before* the `create_dataset` function is applied. Because `time_step=100`, the model skips predicting the first 100 days of the test set, as it lacks the preceding training context. *(Recommended Fix: Define `test_data = scaled_data_array[split - time_step:]`)*.
2. **Autoregressive Error Accumulation:** The 40-day future forecast iteratively feeds the model's own predictions back as inputs. In univariate LSTMs, this causes errors to compound rapidly, resulting in the observed "flattened" prediction curve (converging to a static mean). This is a mathematical artifact, not a realistic market forecast.
3. **The "Naive Forecast" Trap:** LSTMs trained on univariate financial data frequently degenerate into naive forecasters, simply learning to output $P_t \approx P_{t-1}$. The seemingly low Test RMSE should always be benchmarked against a naive baseline (`predictions = actual_prices.shift(1)`) to verify true predictive value.
4. **Lack of Exogenous Features:** The model relies *only* on past closing prices. It ignores trading volume, market sentiment, macroeconomic indicators, or broader market trends, which are essential for robust financial forecasting.

---

## How to Run
1. Open the notebook in **Google Colab**.
2. Run the first cell to install `opendatasets`.
3. Provide your Kaggle username and API key when prompted to download the dataset.
4. Execute all cells sequentially to reproduce the data loading, training, evaluation, and forecasting steps.

---

## Recommended Next Steps for Improvement
- Change the final layer activation to `linear` for standard regression behavior.
- Implement `tf.keras.callbacks.EarlyStopping` to prevent overfitting.
- Add multivariate features (e.g., Volume, Moving Averages, RSI).
- Compare model performance against a naive `shift(1)` baseline to prove actual learning.

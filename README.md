# 📈 Stock Price Prediction using LSTM

## Overview

This project implements a **Long Short-Term Memory (LSTM)** neural network using **TensorFlow/Keras** to predict stock closing prices based on historical market data. The model learns temporal patterns from previous stock prices and forecasts future values using a sequence-based deep learning approach.

The project demonstrates the complete workflow of a time series forecasting task, including data preprocessing, sequence generation, model training, evaluation, visualization, and future prediction.

---

## Dataset

**Source:** Kaggle – Stock Market Dataset

**Dataset Link:**  
https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset

**File Used:** `CCOR.csv`

**Target Variable:** Close Price

---

## Features

- Historical stock price forecasting using LSTM
- Data preprocessing and normalization
- Time-series sequence generation
- Stacked LSTM model implementation
- Model evaluation using RMSE
- Visualization of actual vs predicted prices
- Future stock price forecasting for the next 40 days

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- OpenDatasets

---

## Workflow

### 1. Data Collection

- Download the dataset from Kaggle.
- Load the `CCOR.csv` file.
- Select the **Close** price column for forecasting.

### 2. Data Preprocessing

- Convert the Date column into datetime format.
- Normalize stock prices using MinMaxScaler.
- Split the dataset into training and testing sets.

### 3. Time-Series Dataset Creation

The dataset is transformed into sequential samples using a sliding window approach.

- **Look-back Window:** 100 previous days
- **Prediction Target:** Next day's closing price

### 4. Model Development

The model consists of:

- Input Layer
- LSTM Layer (128 units)
- LSTM Layer (64 units)
- Dense Layer (32 units, ReLU)
- Output Layer (1 unit)

The model is trained using:

- **Optimizer:** Adam
- **Loss Function:** Mean Squared Error (MSE)

### 5. Model Evaluation

After training, the model predicts stock prices on both training and testing datasets.

Performance is evaluated using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

### 6. Visualization

The project compares:

- Original stock prices
- Training predictions
- Testing predictions

using line plots for visual analysis.

### 7. Future Forecasting

The trained LSTM model forecasts the next **40 trading days** by recursively using previous predictions as inputs for future predictions.

---

## Model Architecture

```
Input Layer
      │
      ▼
LSTM (128 Units, return_sequences=True)
      │
      ▼
LSTM (64 Units)
      │
      ▼
Dense (32 Units, ReLU)
      │
      ▼
Dense (1 Unit)
```

---

## Evaluation Metrics

The following metrics are used to evaluate model performance:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## Future Improvements

- Bidirectional LSTM
- GRU-based forecasting
- Attention mechanism
- Transformer models for time series forecasting
- Multivariate stock prediction
- Hyperparameter tuning
- Comparison with ARIMA and Prophet
- Deployment using Streamlit or Flask

---

## Conclusion

This project demonstrates an end-to-end implementation of stock price forecasting using a Long Short-Term Memory (LSTM) neural network. By learning patterns from historical stock prices, the model is able to generate accurate predictions and forecast future values. The project provides a strong foundation for understanding deep learning-based time series forecasting and can be extended with more advanced architectures and additional financial indicators.

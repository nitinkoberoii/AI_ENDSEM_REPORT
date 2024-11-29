import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from datetime import datetime

def download_data(ticker, start_date, end_date):
    """Download historical financial data from Yahoo Finance."""
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    """Preprocess the data by calculating daily returns and handling missing values."""
    data = data[['Adj Close']].dropna()
    data['Returns'] = data['Adj Close'].pct_change()
    data = data.dropna()
    return data

def fit_hmm(data, n_states):
    """Fit a Gaussian Hidden Markov Model to the financial returns data."""
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(data)
    hidden_states = model.predict(data)
    return model, hidden_states

def analyze_hidden_states(model):
    """Analyze the hidden states by extracting means, variances, and transition probabilities."""
    means = model.means_.flatten()
    if model.covariance_type == "full":
        variances = np.array([np.diag(cov) for cov in model.covars_]).flatten()
    elif model.covariance_type == "diag":
        variances = model.covars_.flatten()
    else:
        variances = model.covars_
    return means, variances, model.transmat_

def visualize_hidden_states(data, hidden_states, n_states):
    """Visualize hidden states against stock prices and returns."""
    plt.figure(figsize=(15, 8))
    for i in range(n_states):
        state_data = data[data['Hidden States'] == i]
        plt.plot(state_data.index, state_data['Adj Close'], '.', label=f'State {i}')
    plt.title("Hidden States vs. Adjusted Close Prices")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.legend()
    plt.show()

    plt.figure(figsize=(15, 8))
    plt.plot(data.index, data['Returns'], label="Daily Returns")
    for i in range(n_states):
        state_data = data[data['Hidden States'] == i]
        plt.scatter(state_data.index, state_data['Returns'], label=f'State {i}')
    plt.title("Hidden States vs. Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily Returns")
    plt.legend()
    plt.show()

def predict_future_state(model, recent_data):
    """Predict the likely future state based on the most recent data."""
    return model.predict(recent_data)

if __name__ == "__main__":

    ticker = "AAPL"
    start_date = "2010-01-01"
    end_date = datetime.today().strftime('%Y-%m-%d')
    n_states = 2

    raw_data = download_data(ticker, start_date, end_date)
    processed_data = preprocess_data(raw_data)
    returns_data = processed_data[['Returns']].values

    if np.isnan(returns_data).any():
        print("Data contains NaN values after preprocessing!")
    else:
        print("Data is clean and ready for modeling.")

    hmm_model, hidden_states = fit_hmm(returns_data, n_states)
    means, variances, trans_matrix = analyze_hidden_states(hmm_model)

    print("Means of Hidden States:", means)
    print("Variances of Hidden States:", variances)
    print("Transition Matrix:\n", trans_matrix)

    processed_data['Hidden States'] = hidden_states

    visualize_hidden_states(processed_data, hidden_states, n_states)

    recent_data = returns_data[-1].reshape(-1, 1)
    future_state = predict_future_state(hmm_model, recent_data)
    print("Predicted Future State:", future_state[0])
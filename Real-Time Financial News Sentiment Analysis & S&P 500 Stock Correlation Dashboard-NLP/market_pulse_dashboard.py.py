# %% [markdown]
# # **Market Pulse AI: SPY Sentiment & Forecast Dashboard**
# # (End-to-End Python Script - Corrected)

# %%
# ---------------------------------------------------------------------------
# Section 1: Imports
# ---------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import yfinance as yf
from newsapi import NewsApiClient
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch
from sklearn.preprocessing import MinMaxScaler
# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_loading_spinners
from datetime import datetime, timedelta
import logging
import warnings
import json # For storing data in dcc.Store

# %%
# ---------------------------------------------------------------------------
# Section 2: Configuration & API Keys
# ---------------------------------------------------------------------------

# --- Logging and Warnings ---
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
tf.get_logger().setLevel('ERROR') # Suppress verbose TensorFlow logging

# --- User Configuration ---
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"  # <<< REPLACE WITH YOUR ACTUAL NEWSAPI KEY!
STOCK_TICKER = "SPY"
DATA_FETCH_PERIOD = "2y" # Fetch sufficient data for LSTM training
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS = 25 # Can increase for potentially better accuracy, but longer training
LSTM_BATCH_SIZE = 32

# --- File Paths ---
STOCK_DATA_CSV = f"{STOCK_TICKER}_stock_prices.csv"
NEWS_DATA_CSV = f"{STOCK_TICKER}_news_data.csv"
NEWS_SENTIMENT_CSV = f"{STOCK_TICKER}_news_sentiment.csv"
MERGED_DATA_CSV = f"{STOCK_TICKER}_merged_data.csv"

# --- Model Names ---
FINBERT_MODEL_NAME = "ProsusAI/finbert"
SUMMARY_MODEL_NAME = "distilgpt2" # Smaller, faster GPT-2 model

# --- Sentiment Labels ---
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# %%
# ---------------------------------------------------------------------------
# Section 3: Data Fetching Functions (Corrected)
# ---------------------------------------------------------------------------

def fetch_stock_data(ticker, period, interval="1d", filename=STOCK_DATA_CSV):
    """Fetches historical stock data using yfinance, ensures numeric types, and saves to CSV."""
    logging.info(f"Fetching stock data for {ticker}...")
    try:
        stock_data = yf.download(ticker, period=period, interval=interval, progress=False) # Disable progress bar
        if stock_data.empty:
            logging.error(f"No data fetched for ticker {ticker}. Check ticker symbol or period.")
            return None

        stock_data.reset_index(inplace=True)
        stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.normalize()

        # --- FIX: Ensure numeric types right after download ---
        price_vol_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        for col in price_vol_cols:
            if col in stock_data.columns:
                stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce') # Coerce errors to NaN
            else:
                logging.warning(f"Column '{col}' not found in yfinance download for {ticker}.")

        # Handle potential NaNs from coercion
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_cols:
             if col in stock_data.columns:
                  stock_data[col].fillna(method='ffill', inplace=True) # Forward fill missing prices
                  stock_data[col].fillna(method='bfill', inplace=True) # Backward fill if needed at start
        if 'Volume' in stock_data.columns:
             stock_data['Volume'].fillna(0, inplace=True) # Fill missing volume with 0

        # Drop rows if 'Close' is still NaN after filling (critical column)
        initial_len = len(stock_data)
        stock_data.dropna(subset=['Close'], inplace=True)
        if len(stock_data) < initial_len:
             logging.warning(f"Dropped {initial_len - len(stock_data)} rows with NaN 'Close' prices after filling.")

        if stock_data.empty:
             logging.error("Stock data became empty after handling NaNs.")
             return None
        # ----------------------------------------------------

        stock_data.to_csv(filename, index=False)
        logging.info(f"Stock data saved to {filename}. Shape: {stock_data.shape}. Numeric types verified.")
        return stock_data.copy() # Return a copy
    except Exception as e:
        logging.exception(f"Error fetching stock data for {ticker}: {e}")
        return None

def fetch_news_data(api_key, query, filename=NEWS_DATA_CSV, page_size=100):
    """Fetches financial news using NewsAPI and saves relevant fields to CSV."""
    logging.info(f"Fetching news data for query: '{query}'...")
    if not api_key or api_key == "YOUR_NEWSAPI_KEY":
        logging.error("NewsAPI key is not set. Please add your key to the configuration section.")
        return pd.DataFrame(columns=["publishedAt", "title", "description", "source_name"])

    try:
        newsapi = NewsApiClient(api_key=api_key)
        all_articles = newsapi.get_everything(
            q=query,
            language="en",
            sort_by="publishedAt", # Use publishedAt for more timely news
            page_size=page_size
        )
        if not all_articles or all_articles['status'] != 'ok' or not all_articles['articles']:
            logging.warning(f"No articles found for query '{query}' or API error occurred.")
            return pd.DataFrame(columns=["publishedAt", "title", "description", "source_name"])

        news_df = pd.DataFrame(all_articles["articles"])
        news_df['source_name'] = news_df['source'].apply(lambda x: x.get('name') if isinstance(x, dict) else None)
        news_df = news_df[["publishedAt", "title", "description", "source_name"]]
        news_df.dropna(subset=["title"], inplace=True)
        news_df["publishedAt"] = pd.to_datetime(news_df["publishedAt"]) # Keep timezone info initially
        news_df.to_csv(filename, index=False)
        logging.info(f"News data saved to {filename}. Found {len(news_df)} articles.")
        return news_df.copy()
    except Exception as e:
        logging.exception(f"Error fetching news data: {e}")
        return pd.DataFrame(columns=["publishedAt", "title", "description", "source_name"])

# %%
# ---------------------------------------------------------------------------
# Section 4: Sentiment Analysis Function
# ---------------------------------------------------------------------------

def analyze_sentiment(news_df, model_name=FINBERT_MODEL_NAME, filename=NEWS_SENTIMENT_CSV):
    """Performs sentiment analysis on news titles using a specified FinBERT model."""
    logging.info(f"Performing sentiment analysis using {model_name}...")
    if news_df is None or news_df.empty:
        logging.warning("No news data provided for sentiment analysis.")
        return pd.DataFrame(columns=['publishedAt', 'title', 'description', 'source_name', 'sentiment', 'sentiment_probs'])

    # Ensure required columns exist
    if "title" not in news_df.columns or "publishedAt" not in news_df.columns:
        logging.error("Input news DataFrame missing 'title' or 'publishedAt'. Cannot perform sentiment analysis.")
        # Add missing columns if possible, otherwise return empty
        if "title" not in news_df.columns: news_df["title"] = None
        if "publishedAt" not in news_df.columns: news_df["publishedAt"] = pd.NaT
        news_df["sentiment"] = "Error"
        news_df["sentiment_probs"] = [[0.0, 0.0, 0.0]] * len(news_df)
        return news_df

    try:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        logging.info(f"Sentiment analysis model loaded on {device}.")

        sentiments = []
        probabilities = []

        for title in news_df["title"]:
            # Ensure title is a non-empty string
            if pd.isna(title) or not isinstance(title, str) or not title.strip():
                sentiments.append("Neutral")
                probabilities.append([0.0, 1.0, 0.0])
                continue

            try:
                inputs = tokenizer(str(title), return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)

                probs = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
                label_index = np.argmax(probs)
                label = SENTIMENT_LABELS[label_index]
                sentiments.append(label)
                probabilities.append(probs.tolist())
            except Exception as e_inner:
                logging.error(f"Error analyzing title '{str(title)[:50]}...': {e_inner}")
                sentiments.append("Neutral")
                probabilities.append([0.0, 1.0, 0.0])

        news_df_copy = news_df.copy() # Work on copy
        news_df_copy["sentiment"] = sentiments
        news_df_copy["sentiment_probs"] = probabilities
        news_df_copy.to_csv(filename, index=False)
        logging.info(f"Sentiment analysis complete. Results saved to {filename}")
        return news_df_copy
    except Exception as e:
        logging.exception(f"Error during sentiment analysis setup or processing: {e}")
        news_df_copy = news_df.copy()
        news_df_copy["sentiment"] = "Error"
        news_df_copy["sentiment_probs"] = [[0.0, 0.0, 0.0]] * len(news_df_copy)
        return news_df_copy


# %%
# ---------------------------------------------------------------------------
# Section 5: Data Merging & Preprocessing Function (Corrected)
# ---------------------------------------------------------------------------

def preprocess_and_merge(stock_df, sentiment_df, filename=MERGED_DATA_CSV):
    """Merges stock data with daily aggregated sentiment scores, ensures numeric types, and calculates features."""
    logging.info("Preprocessing and merging stock and sentiment data...")

    if stock_df is None or stock_df.empty:
        logging.error("Stock data is missing or empty. Cannot merge.")
        return None

    stock_df_copy = stock_df.copy()
    stock_df_copy['Date'] = pd.to_datetime(stock_df_copy['Date']).dt.normalize()
    numeric_stock_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Ensure numeric types in stock_df_copy *before* merging
    for col in numeric_stock_cols:
        if col in stock_df_copy.columns:
            stock_df_copy[col] = pd.to_numeric(stock_df_copy[col], errors='coerce')
        else:
            logging.warning(f"Stock data copy missing expected numeric column: {col}")
    stock_df_copy.dropna(subset=['Close'], inplace=True) # Drop rows if 'Close' became NaN

    if sentiment_df is None or sentiment_df.empty:
        logging.warning("Sentiment data is missing or empty. Merging stock data only.")
        merged_df = stock_df_copy
        merged_df['sentiment_score'] = 0.0
        merged_df['news_count'] = 0
    else:
        sentiment_df_copy = sentiment_df.copy()
        try:
            if not all(col in sentiment_df_copy.columns for col in ['publishedAt', 'sentiment_probs', 'title']):
                logging.error("Sentiment DataFrame missing required columns. Merging stock data only.")
                merged_df = stock_df_copy
                merged_df['sentiment_score'] = 0.0
                merged_df['news_count'] = 0
            else:
                # Convert publishedAt to Date (normalize and remove timezone)
                sentiment_df_copy['Date'] = pd.to_datetime(sentiment_df_copy['publishedAt']).dt.tz_convert(None).dt.normalize()
                sentiment_df_copy['sentiment_score_raw'] = sentiment_df_copy['sentiment_probs'].apply(
                    lambda x: x[2] - x[0] if isinstance(x, list) and len(x) == 3 else 0.0
                )
                daily_sentiment = sentiment_df_copy.groupby('Date').agg(
                    sentiment_score=('sentiment_score_raw', 'mean'),
                    news_count=('title', 'count')
                ).reset_index()

                # Merge Data (Left join on stock data)
                merged_df = pd.merge(stock_df_copy, daily_sentiment, on='Date', how='left')

                merged_df['sentiment_score'].fillna(0.0, inplace=True)
                merged_df['news_count'].fillna(0, inplace=True)

        except Exception as e:
            logging.exception(f"Error preparing sentiment data or merging: {e}")
            merged_df = stock_df_copy # Fallback
            merged_df['sentiment_score'] = 0.0
            merged_df['news_count'] = 0

    # --- CRITICAL FIX: Ensure Numeric Types *AFTER* Merging ---
    logging.info("Verifying numeric types in merged data post-merge...")
    final_numeric_cols = [col for col in numeric_stock_cols if col in merged_df.columns]
    for col in final_numeric_cols:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
        if merged_df[col].isnull().any():
             logging.warning(f"NaNs found in column '{col}' after merge/numeric conversion. Attempting fill.")
             if col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                  merged_df[col].fillna(method='ffill', inplace=True)
                  merged_df[col].fillna(method='bfill', inplace=True) # Fill initial NaNs too
             elif col == 'Volume':
                  merged_df[col].fillna(0, inplace=True)
    merged_df.dropna(subset=['Close'], inplace=True) # Final check on critical column
    if merged_df.empty:
        logging.error("Merged dataframe became empty after post-merge cleaning.")
        return None
    logging.info("Post-merge numeric type verification complete.")
    # ---------------------------------------------------------

    # --- Feature Calculation ---
    try:
        # Calculate pct_change safely
        close_shifted = merged_df['Close'].shift(1)
        merged_df['price_change'] = ((merged_df['Close'] - close_shifted) / close_shifted).replace([np.inf, -np.inf], 0.0).fillna(0.0) # Replace inf/-inf with 0, fillna with 0

        # Calculate rolling correlation
        if 'sentiment_score' in merged_df.columns and merged_df['sentiment_score'].notna().sum() >= 3 and merged_df['price_change'].notna().sum() >= 3:
             # Ensure both series used for correlation are float type
             sentiment_scores_numeric = pd.to_numeric(merged_df['sentiment_score'], errors='coerce')
             price_change_numeric = pd.to_numeric(merged_df['price_change'], errors='coerce')
             merged_df['sentiment_price_corr'] = sentiment_scores_numeric.rolling(window=7, min_periods=3).corr(price_change_numeric)
             merged_df['sentiment_price_corr'].fillna(0.0, inplace=True) # Fill initial NaNs in corr with 0
             logging.info("Rolling correlation calculated.")
        else:
             merged_df['sentiment_price_corr'] = 0.0 # Set to 0 if not enough data
             logging.warning("Not enough valid data points to calculate rolling correlation, setting to 0.")

        merged_df.to_csv(filename, index=False)
        logging.info(f"Data merging and feature calculation complete. Shape: {merged_df.shape}. Saved to {filename}")
        return merged_df.copy()

    except Exception as e:
        logging.exception(f"Error during feature calculation after merge: {e}")
        return merged_df.copy() # Return partially processed df

# %%
# ---------------------------------------------------------------------------
# Section 6: LSTM Model Functions (Corrected)
# ---------------------------------------------------------------------------

def create_sequences(data, seq_length):
    """Creates sequences of data for LSTM training."""
    X, y = [], []
    if len(data) <= seq_length:
        logging.warning(f"Data length ({len(data)}) <= sequence length ({seq_length}). Cannot create sequences.")
        return np.array(X), np.array(y)
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    """Builds the LSTM model architecture using tf.keras."""
    tf.keras.backend.clear_session() # Clear previous sessions
    model = Sequential(name="Stock_LSTM_Model")
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape, name="LSTM_1"))
    model.add(Dropout(0.2, name="Dropout_1"))
    model.add(LSTM(units=50, return_sequences=False, name="LSTM_2"))
    model.add(Dropout(0.2, name="Dropout_2"))
    model.add(Dense(units=25, activation='relu', name="Dense_1")) # Added ReLU activation
    model.add(Dense(units=1, name="Output_Dense"))
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info("LSTM model built successfully using tf.keras.")
    # model.summary() # Optional: Print model summary to logs
    return model

def train_and_predict_lstm(data_df, target_column='Close', seq_length=LSTM_SEQUENCE_LENGTH, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE):
    """Trains the LSTM model and generates predictions."""
    logging.info(f"Starting LSTM training for column '{target_column}'...")
    if data_df is None or data_df.empty:
        logging.error("LSTM Error: Input dataframe is None or empty.")
        return None, None, None, None, None, None
    if target_column not in data_df.columns or 'Date' not in data_df.columns:
        logging.error(f"LSTM Error: Required columns ('{target_column}', 'Date') not found.")
        return None, None, None, None, None, None

    try:
        # --- Data Preparation ---
        data_for_lstm = data_df[['Date', target_column]].copy()
        data_for_lstm['Date'] = pd.to_datetime(data_for_lstm['Date']) # Ensure Date is datetime
        data_for_lstm.set_index('Date', inplace=True)
        data_for_lstm.sort_index(inplace=True)

        price_data = data_for_lstm[[target_column]]

        # --- Final Check: Ensure numeric type *before* scaling ---
        if not pd.api.types.is_numeric_dtype(price_data[target_column]):
            logging.warning(f"LSTM Warning: Target column '{target_column}' is not numeric before scaling. Attempting conversion.")
            price_data[target_column] = pd.to_numeric(price_data[target_column], errors='coerce')
            price_data.dropna(subset=[target_column], inplace=True) # Drop rows where conversion failed
            if price_data.empty or not pd.api.types.is_numeric_dtype(price_data[target_column]):
                 logging.error(f"LSTM Error: Target column '{target_column}' could not be converted to numeric. Aborting.")
                 return None, None, None, None, None, None
        # ---

        price_data.fillna(method='ffill', inplace=True) # Fill any remaining NaNs
        price_data.fillna(method='bfill', inplace=True)
        price_data.dropna(inplace=True) # Drop if still NaN

        if len(price_data) <= seq_length:
            logging.error(f"LSTM Error: Not enough data ({len(price_data)} points) after cleaning for sequences of length {seq_length}.")
            return None, None, None, None, None, None

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(price_data)

        X, y = create_sequences(scaled_data, seq_length)

        if len(X) == 0:
             logging.error("LSTM Error: Could not create sequences.")
             return None, None, None, None, None, None

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # --- Correctly Align Test Dates ---
        # y_test corresponds to the original data starting from index: seq_length + split_idx
        test_start_index_in_original = seq_length + split_idx
        if test_start_index_in_original >= len(price_data):
             logging.error("LSTM Error: Calculated test start index is out of bounds.")
             return None, None, None, None, None, None
        test_dates = price_data.index[test_start_index_in_original:]
        # Ensure test_dates length matches y_test length
        if len(test_dates) != len(y_test):
             logging.warning(f"LSTM Warning: Length mismatch between test_dates ({len(test_dates)}) and y_test ({len(y_test)}). Trimming test_dates.")
             test_dates = test_dates[:len(y_test)] # Trim if necessary, though shouldn't happen ideally
        # ---

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # --- Model Building & Training ---
        model = build_lstm_model((X_train.shape[1], 1))
        logging.info(f"Training LSTM model for max {epochs} epochs (Batch Size: {batch_size})...")
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping],
                            verbose=0)
        epochs_ran = len(history.history['loss'])
        logging.info(f"LSTM training finished after {epochs_ran} epochs.")

        # --- Prediction ---
        predicted_scaled = model.predict(X_test)
        predicted_prices = scaler.inverse_transform(predicted_scaled)

        # --- Performance Metrics ---
        train_loss = history.history['loss'][-1]
        val_loss = min(history.history['val_loss']) # Best val loss
        logging.info(f"LSTM final metrics: Train MSE={train_loss:.4f}, Best Val MSE={val_loss:.4f}")

        accuracy = 0.0 # Default
        if not test_dates.empty:
            try:
                # Get actual prices corresponding *exactly* to the predictions
                actual_prices_test = price_data.loc[test_dates, target_column].values
                if len(actual_prices_test) == len(predicted_prices):
                    mask = actual_prices_test != 0
                    if np.any(mask):
                         # Calculate MAPE on non-zero actuals only
                         mape = np.mean(np.abs((actual_prices_test[mask] - predicted_prices.flatten()[mask]) / actual_prices_test[mask])) * 100
                         accuracy = max(0.0, 100 - mape) # Ensure accuracy isn't negative
                    logging.info(f"LSTM Test Accuracy (100-MAPE): {accuracy:.2f}%")
                else:
                    logging.warning("Length mismatch for accuracy calculation after retrieving actual prices.")
            except Exception as acc_e:
                 logging.error(f"Error calculating LSTM accuracy: {acc_e}")
        else:
             logging.warning("No test dates available for accuracy calculation.")

        # Ensure predictions and dates are returned correctly
        if len(predicted_prices) != len(test_dates):
             logging.warning("Final length mismatch between predicted_prices and test_dates. Returning based on predictions length.")
             test_dates = test_dates[:len(predicted_prices)] # Trim dates if needed

        return model, scaler, history, predicted_prices, test_dates, accuracy

    except Exception as e:
        logging.exception(f"Error during LSTM training or prediction: {e}")
        return None, None, None, None, None, None


# %%
# ---------------------------------------------------------------------------
# Section 7: LLM Summary Generation Function (Corrected)
# ---------------------------------------------------------------------------

# Load the text generation pipeline globally
try:
    logging.info(f"Loading text generation model: {SUMMARY_MODEL_NAME}...")
    text_generator = pipeline('text-generation', model=SUMMARY_MODEL_NAME, device=(0 if torch.cuda.is_available() else -1))
    logging.info("Text generation model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load text generation model ({SUMMARY_MODEL_NAME}): {e}. Summary feature disabled.")
    text_generator = None

def generate_financial_summary(df, predicted_prices, test_dates, accuracy):
    """Generates a brief financial summary using a pre-loaded LLM, handling None types."""
    if text_generator is None:
        logging.warning("Text generator model not available.")
        return "AI Summary feature is currently unavailable."
    if df is None or df.empty:
        logging.warning("Summary Error: Source dataframe is missing.")
        return "Summary Error: Missing source data."
    if predicted_prices is None or test_dates is None or len(predicted_prices) == 0 or len(test_dates) == 0:
        logging.warning("Summary Error: Prediction data missing or empty.")
        return "Summary Error: Missing prediction data."
    if not isinstance(accuracy, (int, float)):
         logging.warning(f"Summary Warning: Invalid accuracy value ({accuracy}). Using 0.")
         accuracy = 0.0 # Default if invalid

    logging.info("Generating financial summary...")
    try:
        # --- Extract Key Information ---
        df_sorted = df.sort_values(by='Date', ascending=True)
        if df_sorted.empty:
            logging.error("Cannot generate summary: DataFrame is empty after sorting.")
            return "Summary Error: No data available after sorting."
        latest_data = df_sorted.iloc[-1]

        latest_date_str = latest_data['Date'].strftime('%Y-%m-%d')
        # --- Safe extraction and formatting ---
        latest_close = latest_data.get('Close')
        latest_close_str = f'${latest_close:.2f}' if isinstance(latest_close, (int, float)) else 'N/A'
        latest_sentiment = latest_data.get('sentiment_score')
        latest_sentiment_str = f'{latest_sentiment:.3f}' if isinstance(latest_sentiment, (int, float)) else 'N/A'
        latest_corr = latest_data.get('sentiment_price_corr')
        latest_corr_str = f'{latest_corr:.3f}' if isinstance(latest_corr, (int, float)) else 'N/A'
        accuracy_str = f'{accuracy:.2f}%' if isinstance(accuracy, (int, float)) else 'N/A'
        # ---

        if len(predicted_prices) != len(test_dates):
             logging.warning("Summary Warning: Prediction and test_dates length mismatch.")
             # Attempt to use available data, maybe trim predictions? Or return error?
             min_len = min(len(predicted_prices), len(test_dates))
             if min_len == 0: return "Summary Error: Zero length predictions/dates."
             predicted_prices = predicted_prices[:min_len]
             test_dates = test_dates[:min_len]


        prediction_start_date = test_dates[0].strftime('%Y-%m-%d')
        prediction_end_date = test_dates[-1].strftime('%Y-%m-%d')
        start_pred_price = predicted_prices[0][0] if len(predicted_prices) > 0 else np.nan
        end_pred_price = predicted_prices[-1][0] if len(predicted_prices) > 0 else np.nan

        if pd.isna(start_pred_price) or pd.isna(end_pred_price):
             predicted_trend = "unknown"
             last_predicted_price_str = "N/A"
        else:
            predicted_trend = "upward" if end_pred_price > start_pred_price else "downward" if end_pred_price < start_pred_price else "stable"
            last_predicted_price_str = f'${end_pred_price:.2f}'


        # --- Construct Prompt ---
        prompt = (
            f"Generate a concise financial market outlook summary (3-4 sentences) for {STOCK_TICKER} based on this data:\n"
            f"*   Data as of: {latest_date_str}\n"
            f"*   Latest Closing Price: {latest_close_str}\n"
            f"*   Latest Avg Daily Sentiment: {latest_sentiment_str} (-1 Negative to +1 Positive)\n"
            f"*   Recent Sentiment-Price Correlation: {latest_corr_str} (7-day rolling)\n"
            f"*   LSTM Prediction Accuracy: {accuracy_str} (approx. based on 100-MAPE)\n"
            f"*   Prediction Period: {prediction_start_date} to {prediction_end_date}\n"
            f"*   Predicted Trend: {predicted_trend}\n"
            f"*   Final Predicted Price in Period: {last_predicted_price_str}\n\n"
            f"Focus on the interplay between sentiment and price, and the model's short-term prediction. Avoid definitive financial advice.\n\n"
            f"Summary:"
        )

        # --- Generate Text ---
        generated = text_generator(prompt, max_new_tokens=120, num_return_sequences=1, truncation=True, pad_token_id=text_generator.tokenizer.eos_token_id, do_sample=True, temperature=0.7)
        raw_summary = generated[0]['generated_text']

        # --- Clean up the output ---
        summary_start_index = raw_summary.rfind("Summary:") # Find last occurrence
        if summary_start_index != -1:
            summary = raw_summary[summary_start_index + len("Summary:"):].strip()
            summary = summary.split('\n\n')[0].strip() # Take first paragraph after Summary:
        else:
            summary = raw_summary # Fallback

        if not summary or len(summary) < 10: # Basic check for meaningful summary
             summary = "AI model did not generate a valid summary. Please try again."

        logging.info("Financial summary generated.")
        return summary

    except Exception as e:
        logging.exception(f"Error generating financial summary: {e}")
        return "Error generating summary. Please check application logs."

# %%
# ---------------------------------------------------------------------------
# Section 8: Initial Data Load and Preparation
# ---------------------------------------------------------------------------
logging.info("--- Starting Initial Data Load & Processing ---")

# --- Load/Fetch Stock Data ---
stock_df = None
try:
    stock_df = pd.read_csv(STOCK_DATA_CSV)
    stock_df['Date'] = pd.to_datetime(stock_df['Date'])
    logging.info(f"Loaded existing stock data from {STOCK_DATA_CSV}")
    if stock_df['Date'].max() < pd.Timestamp.now().normalize() - timedelta(days=2):
        logging.info("Stock data seems old. Refetching...")
        stock_df = fetch_stock_data(STOCK_TICKER, DATA_FETCH_PERIOD, filename=STOCK_DATA_CSV)
except FileNotFoundError:
    logging.warning(f"{STOCK_DATA_CSV} not found. Fetching new data...")
    stock_df = fetch_stock_data(STOCK_TICKER, DATA_FETCH_PERIOD, filename=STOCK_DATA_CSV)
except Exception as e:
     logging.exception(f"Error loading/fetching stock data: {e}. Attempting fetch again.")
     stock_df = fetch_stock_data(STOCK_TICKER, DATA_FETCH_PERIOD, filename=STOCK_DATA_CSV)

# --- Load/Fetch/Analyze Sentiment Data ---
news_sentiment_df = None
try:
    news_sentiment_df = pd.read_csv(NEWS_SENTIMENT_CSV)
    news_sentiment_df['publishedAt'] = pd.to_datetime(news_sentiment_df['publishedAt'])
    if 'sentiment_probs' in news_sentiment_df.columns and not news_sentiment_df.empty and isinstance(news_sentiment_df['sentiment_probs'].iloc[0], str):
        try:
            news_sentiment_df['sentiment_probs'] = news_sentiment_df['sentiment_probs'].apply(lambda x: json.loads(x.replace("'", '"')) if isinstance(x, str) else x)
            logging.info("Converted 'sentiment_probs' string to list.")
        except Exception as e_eval:
            logging.warning(f"Could not evaluate 'sentiment_probs' string: {e_eval}. Setting to neutral.")
            news_sentiment_df['sentiment_probs'] = [[0.0, 1.0, 0.0]] * len(news_sentiment_df)
    logging.info(f"Loaded existing sentiment data from {NEWS_SENTIMENT_CSV}")
    if news_sentiment_df.empty or news_sentiment_df['publishedAt'].max().tz_localize(None) < datetime.now().tz_localize(None) - timedelta(days=1):
         logging.info("News data seems old/empty. Refetching/analyzing...")
         news_df_temp = fetch_news_data(NEWSAPI_KEY, f"{STOCK_TICKER} OR S&P 500 OR stock market", filename=NEWS_DATA_CSV)
         news_sentiment_df_new = analyze_sentiment(news_df_temp, filename=NEWS_SENTIMENT_CSV)
         if news_sentiment_df_new is not None and not news_sentiment_df_new.empty:
              news_sentiment_df = news_sentiment_df_new
         else:
              logging.warning("Failed to get new sentiment data. Using old data if available.")
except FileNotFoundError:
    logging.warning(f"{NEWS_SENTIMENT_CSV} not found. Fetching/analyzing new data...")
    news_df_temp = fetch_news_data(NEWSAPI_KEY, f"{STOCK_TICKER} OR S&P 500 OR stock market", filename=NEWS_DATA_CSV)
    news_sentiment_df = analyze_sentiment(news_df_temp, filename=NEWS_SENTIMENT_CSV)
except Exception as e:
     logging.exception(f"Error loading/processing sentiment data: {e}. Attempting fetch/analyze again.")
     news_df_temp = fetch_news_data(NEWSAPI_KEY, f"{STOCK_TICKER} OR S&P 500 OR stock market", filename=NEWS_DATA_CSV)
     news_sentiment_df = analyze_sentiment(news_df_temp, filename=NEWS_SENTIMENT_CSV)

# --- Perform Merging ---
merged_df = preprocess_and_merge(stock_df, news_sentiment_df, filename=MERGED_DATA_CSV)

# --- Perform Initial LSTM Training ---
lstm_model, lstm_scaler, lstm_history = None, None, None
initial_predictions, initial_test_dates, initial_accuracy = None, None, None
if merged_df is not None and not merged_df.empty and 'Close' in merged_df.columns:
    logging.info("Performing initial LSTM training...")
    lstm_model, lstm_scaler, lstm_history, initial_predictions, initial_test_dates, initial_accuracy = train_and_predict_lstm(merged_df)
    if initial_predictions is not None:
        logging.info("Initial LSTM training successful.")
        # Store initial predictions for immediate display
        initial_prediction_store = {
            'predictions': initial_predictions.tolist(),
            'test_dates': initial_test_dates.strftime('%Y-%m-%d').tolist()
        }
        initial_accuracy_store = {'accuracy': initial_accuracy}
    else:
        logging.error("Initial LSTM training failed. Prediction features will be limited.")
        initial_prediction_store = {'predictions': [], 'test_dates': []} # Empty store data
        initial_accuracy_store = {'accuracy': None}
else:
     logging.error("Merged data unavailable or invalid. Skipping initial LSTM training.")
     initial_prediction_store = {'predictions': [], 'test_dates': []}
     initial_accuracy_store = {'accuracy': None}

logging.info("--- Initial Data Load & Processing Finished ---")

# %%
# ---------------------------------------------------------------------------
# Section 9: Dashboard Layout (Corrected & Enhanced)
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[
    'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap'
])
app.title = f"Market Pulse AI: {STOCK_TICKER} Dashboard"

# Define default dates, handling cases where merged_df might be None
min_allowed_date = datetime.now().date() - timedelta(days=365*int(DATA_FETCH_PERIOD[0]))
max_allowed_date = datetime.now().date()
default_end_date = max_allowed_date
default_start_date = default_end_date - timedelta(days=180)

if merged_df is not None and not merged_df.empty and pd.api.types.is_datetime64_any_dtype(merged_df['Date']):
     min_date_data = merged_df['Date'].min().date()
     max_date_data = merged_df['Date'].max().date()
     min_allowed_date = max(min_allowed_date, min_date_data) # Don't allow earlier than actual data
     max_allowed_date = max_date_data
     default_end_date = max_allowed_date
     default_start_date = (default_end_date - timedelta(days=180))
     if default_start_date < min_allowed_date:
          default_start_date = min_allowed_date

# --- Main Layout ---
app.layout = html.Div(style={
    'backgroundColor': '#121212', 'color': '#E0E0E0', 'padding': '30px',
    'fontFamily': '"Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif', 'minHeight': '100vh'
}, children=[
    # --- Header ---
    html.Div([
        html.I(className="fas fa-chart-line", style={'fontSize': '2.5em', 'marginRight': '15px', 'color': '#00ff88'}),
        html.H1(f"📈 Market Pulse AI: {STOCK_TICKER} Sentiment & Forecast Dashboard 📉",
                style={'textAlign': 'center', 'color': '#FFFFFF', 'marginBottom': '5px', 'fontWeight': 'bold', 'display': 'inline-block', 'verticalAlign': 'middle', 'textShadow': '1px 1px 3px #00ff88'}),
    ], style={'textAlign': 'center', 'marginBottom': '15px'}),
    html.Div("Leveraging NLP & Deep Learning for Market Insights",
             style={'textAlign': 'center', 'fontSize': '1.2em', 'marginBottom': '35px', 'color': '#00bbff', 'fontStyle': 'italic'}),

    # --- Controls Row ---
    html.Div([
         html.Div([
            html.Label("Select Date Range:", style={'marginRight': '10px', 'fontWeight': 'bold', 'color': '#FFFFFF'}),
            dcc.DatePickerRange(
                id='date-range-picker', min_date_allowed=min_allowed_date, max_date_allowed=max_allowed_date,
                start_date=default_start_date, end_date=default_end_date, display_format='YYYY-MM-DD',
                style={'marginRight': '30px', 'fontSize': '14px', 'border': '1px solid #555'}
            ),
        ], style={'display': 'inline-block', 'verticalAlign': 'middle'}),
        html.Button('Train LSTM & Predict', id='run-lstm-button', n_clicks=0, className="control-button train-button", style={'fontWeight': 'bold'}),
        html.Button('Update Sentiment Data', id='update-sentiment-button', n_clicks=0, className="control-button update-button", style={'fontWeight': 'bold'}),
        html.Button('Generate AI Summary', id='generate-summary-button', n_clicks=0, className="control-button summary-button", style={'fontWeight': 'bold', 'display': 'inline-block' if text_generator else 'none'}),
    ], style={'textAlign': 'center', 'marginBottom': '30px', 'padding': '15px', 'backgroundColor': 'rgba(42, 42, 42, 0.8)', 'borderRadius': '10px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.3)'}),

    # --- Metrics Row ---
    html.Div([
        html.Div([
            html.H4("LSTM Model Performance", style={'color': '#00ff88', 'borderBottom': '2px solid #00ff88', 'paddingBottom': '8px', 'marginBottom': '18px'}),
            dash_loading_spinners.Pulse(html.P(id='lstm-metrics-output', children="Train model to see metrics.", style={'fontSize': '15px', 'lineHeight': '1.7', 'minHeight': '60px'})),
        ], className="metric-card"),
        html.Div([
            html.H4("Sentiment Analysis Insights", style={'color': '#00bbff', 'borderBottom': '2px solid #00bbff', 'paddingBottom': '8px', 'marginBottom': '18px'}),
            dash_loading_spinners.Pulse(html.P(id='sentiment-metrics-output', children="Update sentiment to see metrics.", style={'fontSize': '15px', 'lineHeight': '1.7', 'minHeight': '60px'})),
        ], className="metric-card"),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'gap': '25px', 'marginBottom': '30px'}),

    # --- AI Summary Row ---
    html.Div([
         html.Div([
            html.H4("AI Generated Market Summary", style={'color': '#ff9900', 'borderBottom': '2px solid #ff9900', 'paddingBottom': '8px', 'marginBottom': '18px'}),
            dash_loading_spinners.Pulse(dcc.Markdown(id='ai-summary-output', children="Click 'Generate AI Summary' for insights.", style={'lineHeight': '1.7', 'fontSize': '15px', 'minHeight': '80px'})),
        ], className="metric-card", style={'maxWidth': '85%', 'margin': '10px auto'}),
    ], style={'marginBottom': '30px', 'display': 'block' if text_generator else 'none'}),

    # --- Graphs Row ---
    html.Div([
        html.Div(dash_loading_spinners.Graph(dcc.Graph(id='price-prediction-graph', style={'height': '450px'})), className="graph-card"),
        html.Div(dash_loading_spinners.Graph(dcc.Graph(id='sentiment-trend-graph', style={'height': '450px'})), className="graph-card"),
    ], className="graph-row"),

    # --- Correlation Graph ---
    html.Div([
        html.Div(dash_loading_spinners.Graph(dcc.Graph(id='correlation-trend-graph', style={'height': '400px'})), className="graph-card", style={'flexBasis': '98%'}),
    ], className="graph-row", style={'justifyContent': 'center'}),

    # --- Hidden Storage (using initial data as default) ---
    dcc.Store(id='prediction-store', storage_type='memory', data=initial_prediction_store),
    dcc.Store(id='accuracy-store', storage_type='memory', data=initial_accuracy_store),

    html.Footer("Market Pulse AI Dashboard | Powered by NLP & Deep Learning", style={'textAlign': 'center', 'marginTop': '50px', 'paddingTop': '20px', 'fontSize': '13px', 'color': '#888', 'borderTop': '1px solid #333'})
])

# --- CSS Styling via index_string ---
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <!-- Add Font Awesome for Icons -->
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
        <style>
            body { margin: 0; background-color: #121212; } /* Ensure body bg matches */
            .control-button {
                background-color: #00ff88; color: black; border: none;
                padding: 10px 18px; margin: 8px; border-radius: 5px;
                cursor: pointer; transition: background-color 0.3s ease, transform 0.1s ease, box-shadow 0.2s ease;
                font-size: 14px; box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }
            .control-button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.4); }
            .control-button:active { transform: translateY(0px); box-shadow: 0 2px 4px rgba(0,0,0,0.3); }
            .train-button { background-color: #00ff88; } .train-button:hover { background-color: #33ff99; }
            .update-button { background-color: #00bbff; } .update-button:hover { background-color: #33ccff; }
            .summary-button { background-color: #ff9900; } .summary-button:hover { background-color: #ffad33; }

            .metric-card {
                background-color: #1e1e1e; padding: 25px; border-radius: 12px;
                flex: 1; margin: 10px; min-width: 320px;
                box-shadow: 0 6px 12px rgba(0,0,0,0.5);
                border-left: 5px solid #444; transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
             .metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 16px rgba(0,0,0,0.6); }
            .metric-card:nth-child(1) { border-left-color: #00ff88; }
            .metric-card:nth-child(2) { border-left-color: #00bbff; }
            /* Specific style for summary card */
            .metric-card:has(#ai-summary-output) { border-left-color: #ff9900; }
            .metric-card h4 { margin-top: 0; font-weight: 500; }

            .graph-row {
                display: flex; flex-wrap: wrap;
                justify-content: space-between; margin-bottom: 30px; gap: 30px;
            }
            .graph-card {
                background-color: rgba(30, 30, 30, 0.9); /* Darker, slightly transparent */
                padding: 20px; border-radius: 10px;
                flex: 1 1 48%;
                box-shadow: 0 5px 15px rgba(0,0,0,0.4);
                border: 1px solid #333; /* Subtle border */
            }
        </style>
    </head>
    <body>
        <div id="react-entry-point">
            <div class="_dash-loading"> <!-- Optional: You can style a loading message here -->
                Loading Dashboard...
            </div>
        </div>
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# %%
# ---------------------------------------------------------------------------
# Section 10: Dashboard Callbacks (Corrected)
# ---------------------------------------------------------------------------

# --- Callback to Update Graphs based on Date Range & Predictions ---
@app.callback(
    [Output('price-prediction-graph', 'figure'),
     Output('sentiment-trend-graph', 'figure'),
     Output('correlation-trend-graph', 'figure')],
    [Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date'),
     Input('prediction-store', 'data')] # Listen to prediction updates from the store
)
def update_main_graphs(start_date_str, end_date_str, prediction_data_json):
    logging.info("Callback triggered: update_main_graphs")
    # ... (definition of empty_layout and create_empty_figure remains the same) ...
    empty_layout = {'template': 'plotly_dark', 'plot_bgcolor': '#1e1e1e', 'paper_bgcolor': '#121212',
                    'xaxis': {'showgrid': True, 'gridcolor': '#333'}, 'yaxis': {'showgrid': True, 'gridcolor': '#333'}, 'font': {'color': '#E0E0E0'}}
    def create_empty_figure(title):
        fig = go.Figure(layout=empty_layout)
        fig.update_layout(title=dict(text=title, x=0.5, font=dict(size=16)), annotations=[dict(text="No data available", xref="paper", yref="paper", showarrow=False, font=dict(size=16))])
        return fig

    global merged_df
    if merged_df is None or merged_df.empty:
         logging.warning("Graph Update: Merged data not available.")
         return create_empty_figure('Stock Price Data Unavailable'), create_empty_figure('Sentiment Data Unavailable'), create_empty_figure('Correlation Data Unavailable')

    # --- Filter Data ---
    try:
        if not pd.api.types.is_datetime64_any_dtype(merged_df['Date']):
             merged_df['Date'] = pd.to_datetime(merged_df['Date'])
        start_date = pd.to_datetime(start_date_str).normalize()
        end_date = pd.to_datetime(end_date_str).normalize()
        # Use boolean indexing for robust filtering
        mask = (merged_df['Date'] >= start_date) & (merged_df['Date'] <= end_date)
        filtered_df = merged_df.loc[mask].copy() # Use copy

    except Exception as e:
        logging.error(f"Graph Update Error: Filtering data by date range ('{start_date_str}' to '{end_date_str}'): {e}")
        return create_empty_figure(f'Error processing date range'), create_empty_figure('Error processing date range'), create_empty_figure('Error processing date range')

    if filtered_df.empty:
        logging.warning("Graph Update: No data available for the selected date range.")
        return create_empty_figure('No Stock Price Data in Range'), create_empty_figure('No Sentiment Data in Range'), create_empty_figure('No Correlation Data in Range')

    # --- Price Graph ---
    fig_price = go.Figure(layout=empty_layout)
    if 'Close' in filtered_df.columns:
        fig_price.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['Close'], mode='lines', name='Actual Close', line=dict(color='#00ff88', width=2)))
    else: logging.warning("Price Graph: 'Close' column missing.")

    # Add predictions
    prediction_data = prediction_data_json # Already deserialized by Dash
    if prediction_data and isinstance(prediction_data, dict) and 'predictions' in prediction_data and 'test_dates' in prediction_data:
        try:
            preds_list = prediction_data['predictions']
            dates_list = prediction_data['test_dates']
            if dates_list and preds_list: # Check if lists are not empty
                pred_dates = pd.to_datetime(dates_list)
                preds = np.array(preds_list).flatten()
                if len(preds) == len(pred_dates):
                    pred_df = pd.DataFrame({'Date': pred_dates, 'Predicted': preds}).set_index('Date')
                    # Filter predictions to match the main graph's date range
                    filtered_preds = pred_df.loc[start_date:end_date]
                    if not filtered_preds.empty:
                         fig_price.add_trace(go.Scatter(x=filtered_preds.index, y=filtered_preds['Predicted'], mode='lines', name='Predicted', line=dict(color='#ff4d4d', width=2, dash='dot'))) # Changed color/style
                else: logging.warning("Prediction plot warning: Length mismatch.")
            # else: logging.info("Prediction data in store is empty or invalid.")
        except Exception as e_pred: logging.error(f"Error processing prediction data: {e_pred}")

    fig_price.update_layout(title=f'{STOCK_TICKER} Price & LSTM Prediction', xaxis_title=None, yaxis_title='Price (USD)', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # --- Sentiment Trend Graph ---
    fig_sent = go.Figure(layout=empty_layout)
    if 'sentiment_score' in filtered_df.columns:
        fig_sent.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['sentiment_score'], mode='lines', name='Avg Daily Sentiment', line=dict(color='#00bbff', width=2), fill='tozeroy'))
        fig_sent.update_layout(title='Daily News Sentiment Trend', xaxis_title=None, yaxis_title='Sentiment Score (-1 to 1)')
    else:
        logging.warning("Sentiment Graph: 'sentiment_score' column missing.")
        fig_sent = create_empty_figure('Sentiment Data Unavailable')

    # --- Correlation Trend Graph ---
    fig_corr = go.Figure(layout=empty_layout)
    if 'sentiment_price_corr' in filtered_df.columns:
        if filtered_df['sentiment_price_corr'].notna().any():
            fig_corr.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df['sentiment_price_corr'], mode='lines', name='7-Day Rolling Correlation', line=dict(color='#ff9900', width=2)))
            fig_corr.update_layout(title='Sentiment vs. Price Change Correlation (7d Roll)', xaxis_title=None, yaxis_title='Correlation')
        else: fig_corr = create_empty_figure('Correlation Data Not Available for Range')
    else:
        logging.warning("Correlation Graph: 'sentiment_price_corr' column missing.")
        fig_corr = create_empty_figure('Correlation Data Unavailable')

    return fig_price, fig_sent, fig_corr

# --- Callback to Run LSTM Training ---
@app.callback(
    [Output('lstm-metrics-output', 'children'),
     Output('prediction-store', 'data', allow_duplicate=True),
     Output('accuracy-store', 'data', allow_duplicate=True)],
    [Input('run-lstm-button', 'n_clicks')],
    prevent_initial_call=True
)
def run_lstm_training_callback(n_clicks):
    # ... (same logic as before, ensure global merged_df is accessed) ...
    triggered_id = callback_context.triggered_id
    if not triggered_id or triggered_id != 'run-lstm-button.n_clicks':
        raise dash.exceptions.PreventUpdate

    logging.info(f"Callback triggered: run_lstm_training (Click {n_clicks})")
    global merged_df, lstm_model, lstm_scaler, lstm_history, initial_predictions, initial_test_dates, initial_accuracy

    if merged_df is None or merged_df.empty:
        logging.error("LSTM Callback Error: Merged data is unavailable.")
        return "Error: Merged data unavailable.", dash.no_update, dash.no_update

    logging.info("Re-training LSTM model via callback...")
    lstm_model_new, scaler_new, history_new, predictions_new, test_dates_new, accuracy_new = train_and_predict_lstm(
        merged_df, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE
    )

    if predictions_new is not None and history_new is not None and test_dates_new is not None and accuracy_new is not None:
        # Update global state (use with caution in multi-user scenarios)
        lstm_model, lstm_scaler, lstm_history = lstm_model_new, scaler_new, history_new
        initial_predictions, initial_test_dates, initial_accuracy = predictions_new, test_dates_new, accuracy_new

        train_loss = history_new.history['loss'][-1]
        val_loss = min(history_new.history['val_loss'])
        metrics_text = f"Training Complete!\nTrain MSE: {train_loss:.4f} | Best Val MSE: {val_loss:.4f} | Test Accuracy: {accuracy_new:.2f}%"
        logging.info(f"LSTM Re-training successful. Metrics: {metrics_text}")

        prediction_data_store = {
            'predictions': predictions_new.tolist(),
            'test_dates': test_dates_new.strftime('%Y-%m-%d').tolist()
        }
        accuracy_store_data = {'accuracy': accuracy_new}

        return metrics_text, prediction_data_store, accuracy_store_data
    else:
        logging.error("LSTM Callback Error: Re-training failed.")
        error_text = "LSTM training failed. Check logs for details."
        # Return error message and clear stores or keep old data? Clearing seems safer.
        return error_text, {'predictions': [], 'test_dates': []}, {'accuracy': None}

# --- Callback to Update Sentiment Data ---
@app.callback(
    [Output('sentiment-metrics-output', 'children'),
     Output('prediction-store', 'data', allow_duplicate=True), # Reset predictions
     Output('accuracy-store', 'data', allow_duplicate=True)], # Reset accuracy
    [Input('update-sentiment-button', 'n_clicks')],
     prevent_initial_call=True
)
def update_sentiment_data_callback(n_clicks):
    # ... (same logic as before, accessing/updating globals news_sentiment_df, merged_df, stock_df) ...
    triggered_id = callback_context.triggered_id
    if not triggered_id or triggered_id != 'update-sentiment-button.n_clicks':
        raise dash.exceptions.PreventUpdate

    logging.info(f"Callback triggered: update_sentiment_data (Click {n_clicks})")
    global news_sentiment_df, merged_df, stock_df

    # Reset stores immediately to indicate data is changing
    reset_prediction_data = {'predictions': [], 'test_dates': []}
    reset_accuracy_data = {'accuracy': None}

    # --- Fetch and Analyze New Sentiment ---
    news_df_new = fetch_news_data(NEWSAPI_KEY, f"{STOCK_TICKER} OR S&P 500 OR stock market", filename=NEWS_DATA_CSV)
    if news_df_new is None or news_df_new.empty:
        logging.warning("Sentiment Update Warning: Failed to fetch new news data.")
        # Keep existing metrics and stores
        return "Failed to fetch new news data. Metrics unchanged.", dash.no_update, dash.no_update

    sentiment_df_new = analyze_sentiment(news_df_new, filename=NEWS_SENTIMENT_CSV)
    if sentiment_df_new is None or sentiment_df_new.empty:
         logging.error("Sentiment Update Error: Sentiment analysis failed.")
         return "Failed to analyze sentiment of new data. Metrics unchanged.", dash.no_update, dash.no_update

    # --- Update Global DataFrames and Re-Merge ---
    news_sentiment_df = sentiment_df_new # Update global sentiment df
    if stock_df is not None:
        merged_df_new = preprocess_and_merge(stock_df, news_sentiment_df, filename=MERGED_DATA_CSV)
        if merged_df_new is not None:
            merged_df = merged_df_new # Update global merged df
            logging.info("Sentiment Update: Data successfully updated and re-merged.")
            sentiment_text = "Sentiment data updated. Re-run LSTM for new predictions based on latest data."
        else:
            logging.error("Sentiment Update Error: Failed to re-merge data.")
            sentiment_text = "Sentiment update done, but merging failed. Metrics may be outdated."
            # Don't reset stores if merge failed, keep old predictions? Or reset? Resetting is safer.
            # reset_prediction_data = dash.no_update
            # reset_accuracy_data = dash.no_update
    else:
         logging.error("Sentiment Update Error: Stock data not available for re-merging.")
         sentiment_text = "Sentiment updated, but stock data missing for re-merge."
         # reset_prediction_data = dash.no_update
         # reset_accuracy_data = dash.no_update

    # --- Calculate and display metrics based on the newly updated 'merged_df' ---
    if merged_df is not None and not merged_df.empty:
        try:
            latest_corr = merged_df['sentiment_price_corr'].iloc[-1] if 'sentiment_price_corr' in merged_df.columns and not merged_df.empty else np.nan
            avg_sentiment = merged_df['sentiment_score'].mean() if 'sentiment_score' in merged_df.columns else np.nan

            sentiment_text += f"\nLatest 7d Corr: {'N/A' if pd.isna(latest_corr) else f'{latest_corr:.3f}'}"
            sentiment_text += f" | Overall Avg Sentiment: {'N/A' if pd.isna(avg_sentiment) else f'{avg_sentiment:.3f}'}"
        except IndexError: sentiment_text += "\nCould not retrieve latest metrics."
        except Exception as e_metrics: logging.error(f"Error calculating sentiment metrics: {e_metrics}")
    else: sentiment_text += "\nMerged data unavailable for metrics."

    return sentiment_text, reset_prediction_data, reset_accuracy_data


# --- Callback to Generate AI Summary ---
@app.callback(
    Output('ai-summary-output', 'children'),
    [Input('generate-summary-button', 'n_clicks')],
    [State('prediction-store', 'data'),
     State('accuracy-store', 'data')],
    prevent_initial_call=True
)
def generate_ai_summary_callback(n_clicks, prediction_data_json, accuracy_data_json):
    # ... (same logic as before, accessing global merged_df) ...
    triggered_id = callback_context.triggered_id
    if not triggered_id or triggered_id != 'generate-summary-button.n_clicks':
        raise dash.exceptions.PreventUpdate

    logging.info("Callback triggered: generate_ai_summary")
    global merged_df # Need latest merged data

    if merged_df is None or merged_df.empty:
        logging.warning("Summary Callback: Merged data unavailable.")
        return "Error: Merged data needed for summary."
    if not text_generator:
         logging.warning("Summary Callback: Text generator unavailable.")
         return "Error: AI Summary feature unavailable."

    # --- Get Prediction Data and Accuracy from Store ---
    preds_to_use, dates_to_use, acc_to_use = None, None, None
    prediction_data = prediction_data_json # Already dict
    accuracy_data = accuracy_data_json # Already dict

    if prediction_data and accuracy_data and prediction_data.get('predictions') and prediction_data.get('test_dates') and accuracy_data.get('accuracy') is not None:
        logging.info("Using prediction data from store for summary.")
        try:
            preds_to_use = np.array(prediction_data['predictions']).reshape(-1, 1)
            dates_to_use = pd.to_datetime(prediction_data['test_dates'])
            acc_to_use = accuracy_data['accuracy']
        except Exception as e_conv:
             logging.error(f"Error converting data from store for summary: {e_conv}")
             return "Error processing stored prediction data."
    else:
        # Fallback to initial data if store is empty/invalid
        logging.info("Prediction store empty/invalid, trying initial data for summary.")
        if initial_predictions is not None and initial_test_dates is not None and initial_accuracy is not None:
            preds_to_use = initial_predictions
            dates_to_use = initial_test_dates
            acc_to_use = initial_accuracy
        else:
             logging.warning("No valid prediction data found (store or initial). Cannot generate summary.")
             return "Error: Prediction data unavailable. Please run LSTM training."

    # --- Generate Summary ---
    if preds_to_use is None or dates_to_use is None or acc_to_use is None:
         logging.error("Valid prediction/accuracy data could not be retrieved.")
         return "Error retrieving necessary data for summary."

    summary = generate_financial_summary(merged_df, preds_to_use, dates_to_use, acc_to_use)
    return summary


# %%
# ---------------------------------------------------------------------------
# Section 11: Run the Dashboard
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    port = 8050 # Default port
    print(f"--- Starting Market Pulse AI Dashboard on http://127.0.0.1:{port}/ ---")
    try:
        # Set debug=False for production/deployment
        # Set debug=True for development (provides in-browser error messages)
        app.run_server(debug=True, port=port)
    except OSError as e:
         if "address already in use" in str(e).lower():
              print(f"\nERROR: Port {port} is already in use.")
              print("Try closing the process using the port or choose a different port.")
         else:
              print(f"\nAn OS error occurred: {e}")
    except Exception as e:
        print(f"\nError starting Dash server: {e}")
        logging.exception("Dash server failed to start.")
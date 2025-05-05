import os
from dotenv import load_dotenv
import logging

# --- Load Environment Variables ---
load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

# --- General Configuration ---
STOCK_TICKER = "SPY"
DATA_FETCH_PERIOD = "2y" # Sufficient data for LSTM

# --- File Paths (Relative to project root, assuming app runs from root or paths adjusted) ---
# Adjust paths if app runs from src/
# DATA_DIR = "../data" # If running from src/
# MODEL_DIR = "../models" # If running from src/
DATA_DIR = "data" # If running from project root
MODEL_DIR = "models" # If running from project root
STOCK_DATA_CSV = os.path.join(DATA_DIR, f"{STOCK_TICKER}_stock_prices.csv")
NEWS_DATA_CSV = os.path.join(DATA_DIR, f"{STOCK_TICKER}_news_data.csv")
NEWS_SENTIMENT_CSV = os.path.join(DATA_DIR, f"{STOCK_TICKER}_news_sentiment.csv")
MERGED_DATA_CSV = os.path.join(DATA_DIR, f"{STOCK_TICKER}_merged_data.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "lstm_model.keras") # Use .keras extension
SCALER_FILE = os.path.join(MODEL_DIR, "lstm_scaler.pkl")


# --- Model Configuration ---
FINBERT_MODEL_NAME = "ProsusAI/finbert"
SUMMARY_MODEL_NAME = "distilgpt2" # Smaller, faster GPT-2 model
LSTM_SEQUENCE_LENGTH = 60
LSTM_EPOCHS = 25
LSTM_BATCH_SIZE = 32
LSTM_TARGET_COLUMN = 'Close' # Explicitly define the target column

# --- Sentiment Labels ---
SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# --- API Key Check ---
if not NEWSAPI_KEY:
    logging.error("CRITICAL: NewsAPI key not found in .env file or environment variables.")
    # Consider raising ValueError or handling this gracefully in functions that need it
    NEWSAPI_KEY = None # Allows app to run, but news features will be disabled
    logging.warning("Proceeding without NewsAPI key. Sentiment analysis features will be disabled.")

# --- Environment Settings (Moved from main script) ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Suppress excessive TensorFlow logging
# Set GPU visibility based on environment variable or default to disable
CUDA_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "-1")
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_DEVICES
if CUDA_DEVICES == "-1":
    logging.info("TensorFlow GPU usage explicitly disabled via config.")
else:
    logging.info(f"TensorFlow CUDA_VISIBLE_DEVICES set to: {CUDA_DEVICES}")

# Optional TF config (can cause issues on some systems)
# import tensorflow as tf
# try:
#     if CUDA_DEVICES == "-1":
#         tf.config.set_visible_devices([], 'GPU')
#     physical_devices_gpu = tf.config.list_physical_devices('GPU')
#     logging.info(f"TensorFlow GPUs available: {physical_devices_gpu}")
#     if physical_devices_gpu:
#         # Configure memory growth if needed
#         try:
#             for gpu in physical_devices_gpu:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#             logging.info("Enabled memory growth for GPUs")
#         except RuntimeError as e:
#             logging.error(f"Could not set memory growth for GPU: {e}")
# except Exception as e:
#      logging.error(f"Error configuring TensorFlow devices: {e}")

# --- Text Generation Pipeline Initialization (centralized) ---
# Initialize here so it's loaded once
TEXT_GENERATOR = None
try:
    from transformers import pipeline as hf_pipeline
    # Force CPU for text generation if needed, especially if TF is using GPU for LSTM
    TEXT_GENERATOR = hf_pipeline('text-generation', model=SUMMARY_MODEL_NAME, device='cpu')
    logging.info(f"Loading text generation model: {SUMMARY_MODEL_NAME} on CPU...")
except Exception as e:
    logging.error(f"Failed to load text generation model ({SUMMARY_MODEL_NAME}): {e}")
    logging.warning("Text generation features will be disabled")
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ensure dirs
for d in (DATA_DIR, RAW_DIR, PROC_DIR, MODELS_DIR):
    os.makedirs(d, exist_ok=True)

# Sentiment model config
HF_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"  # example
BATCH_SIZE = 32  # for HF batching

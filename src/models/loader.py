from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
LOCAL_DIR = "models/pretrained"


def load_pretrained_model():
    """
    Загружает предобученную модель в локальную папку.
    Если она уже есть — использует локальную копию.
    """

    os.makedirs(LOCAL_DIR, exist_ok=True)

    print(f"Загрузка модели {MODEL_NAME}...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=LOCAL_DIR
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        cache_dir=LOCAL_DIR
    )

    print(f"Модель успешно загружена → {LOCAL_DIR}")
    return tokenizer, model

import re
import json
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

# Скачиваем стоп-слова NLTK, если они отсутствуют
try:
    stopwords.words('russian')
except LookupError:
    try:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords')
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading NLTK stopwords: {e}")
        print("Please check your internet connection. If the problem persists,")
        print("you can try downloading them manually by running this command in your terminal:")
        print("python -m nltk.downloader stopwords")


class WordCloudGenerator:
    def __init__(self, extra_stopwords=None):
        self.extra_stopwords = extra_stopwords or set()

    def _clean_text(self, text):
        # 1. Удаление ссылок
        text = re.sub(r"http\S+|www\S+", " ", text)

        # 2. Удаление спецсимволов, эмодзи и пунктуации
        text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", " ", text)

        # 3. Удаление лишних пробелов
        text = re.sub(r"\s+", " ", text).strip()

        # 4. Подготовка стоп-слов
        stop_words = set(stopwords.words("russian")) | set(stopwords.words("english"))
        stop_words |= self.extra_stopwords

        # 5. Фильтрация стоп-слов
        cleaned = " ".join(
            word for word in text.split()
            if word.lower() not in stop_words
        )

        return cleaned

    def generate_wordcloud(self, json_file, output_file, max_words=200, background_color='white', width=800, height=400):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list) and data and 'text' in data[0]:
            text = " ".join([item['text'] for item in data])
        elif isinstance(data, dict) and 'comments' in data:
            text = " ".join([item['text'] for item in data['comments']])
        else:
            all_texts = []
            def find_text(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if key == 'text' and isinstance(value, str):
                            all_texts.append(value)
                        else:
                            find_text(value)
                elif isinstance(obj, list):
                    for item in obj:
                        find_text(item)
            find_text(data)
            text = " ".join(all_texts)

        if not text:
            print(f"Warning: No text found in {json_file}. Cannot generate word cloud.")
            return

        cleaned_text = self._clean_text(text)

        if not cleaned_text:
            print(f"Warning: Text in {json_file} was empty after cleaning. Cannot generate word cloud.")
            return

        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=max_words
        ).generate(cleaned_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"WordCloud сохранён как {output_file}")
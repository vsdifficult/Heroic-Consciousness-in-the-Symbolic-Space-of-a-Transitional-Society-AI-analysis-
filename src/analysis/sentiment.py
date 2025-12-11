from transformers import AutoModel, AutoTokenizer 
import torch, json
from typing import List, Dict 
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

class SentimentAnalyzer:
    """Базовый класс для анализа тональности"""
    
    def __init__(self):
        self.model_name = 'Tochka-AI/ruRoPEBert-classic-base-2k'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            attn_implementation='eager'
        )  
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def predict_sentiment(self, texts: List[str]) -> List[float]:
        """Предсказывает тональность для списка текстов"""
        sentiment_out = []
        for text in texts:
            with torch.no_grad():
                inputs = self.tokenizer(
                    text, 
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True,
                    max_length=512
                ).to(self.model.device) 
                proba = torch.sigmoid(self.model(**inputs).logits).cpu().numpy()[0] 
                sentiment_out.append(float(proba))
        return sentiment_out
    
    def ensemble_filter(self, data: List[float], n_filters: int = 100, 
                       polyorder: int = 0, **savgol_args) -> List[float]:
        """Ансамблевая фильтрация данных"""
        if len(data) < 10:
            return data
            
        filt = 0
        start = max(3, len(data) // 10)
        stop = len(data) // 4
        step = max(1, (stop - start) // n_filters)
        
        for window_size in range(start, stop, step):
            if window_size % 2 == 0:
                window_size += 1
            if window_size > len(data):
                window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
            if window_size < 3:
                window_size = 3
                
            res = savgol_filter(
                data, 
                window_length=window_size, 
                polyorder=polyorder, 
                **savgol_args
            )
            filt += res
        return (filt / n_filters).tolist()


class YouTubeSentimentAnalyzer(SentimentAnalyzer):
    """Класс для анализа тональности комментариев из JSON"""
    
    def __init__(self):
        super().__init__()

    def load_comments_from_json(self, json_file: str) -> List[Dict]:
        """Загружает комментарии из JSON файла"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('comments', [])

    def analyze_from_json(self, json_file: str, output_plot: str = 'sentiment_analysis.png',
                         use_ensemble: bool = False) -> Dict:
        """
        Анализирует тональность комментариев из JSON и сохраняет график
        
        Args:
            json_file: путь к JSON файлу с комментариями
            output_plot: путь для сохранения графика
            use_ensemble: использовать ансамблевую фильтрацию
            
        Returns:
            Dict: результаты анализа
        """
        print(f"Загружаю комментарии из {json_file}...")
        comments = self.load_comments_from_json(json_file)
        
        if not comments:
            print("Нет комментариев для анализа")
            return {}
        
        print(f"Найдено {len(comments)} комментариев")
        
        # Извлекаем тексты
        texts = [c.get('text', '') for c in comments if c.get('text')]
        
        if not texts:
            print("Нет текстов для анализа")
            return {}
        
        print(f"Анализирую тональность {len(texts)} комментариев...")
        sentiments = self.predict_sentiment(texts)
        
    # Применяем фильтрациюz
        if use_ensemble and len(sentiments) > 10:
            print("Применяю ансамблевую фильтрацию...")
            filtered_sentiments = self.ensemble_filter(sentiments)
        else:
            window_length = max(3, len(sentiments) // 15)
            if window_length % 2 == 0:
                window_length += 1
            if window_length <= len(sentiments):
                filtered_sentiments = savgol_filter(
                    sentiments, 
                    window_length=window_length, 
                    polyorder=0
                )
            else:
                filtered_sentiments = sentiments
        
        # Создаем график
        plt.figure(figsize=(12, 6), dpi=300)
        plt.plot(sentiments, alpha=0.3, label='Исходные данные')
        plt.plot(filtered_sentiments, linewidth=2, label='Сглаженные данные')
        plt.xlabel('Номер комментария')
        plt.ylabel('Тональность')
        plt.title(f'Анализ тональности комментариев (всего: {len(sentiments)})')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()
        
        print(f"График сохранен в {output_plot}")
        
        # Статистика
        results = {
            "total_comments": len(comments),
            "analyzed_comments": len(sentiments),
            "average_sentiment": float(sum(sentiments) / len(sentiments)),
            "min_sentiment": float(min(sentiments)),
            "max_sentiment": float(max(sentiments)),
            "output_plot": output_plot
        }
        
        print(f"\nСредняя тональность: {results['average_sentiment']:.3f}")
        print(f"Мин тональность: {results['min_sentiment']:.3f}")
        print(f"Макс тональность: {results['max_sentiment']:.3f}")
        
        return results

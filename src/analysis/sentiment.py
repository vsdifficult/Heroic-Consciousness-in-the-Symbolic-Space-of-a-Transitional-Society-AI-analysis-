from transformers import AutoModelForSequenceClassification, AutoTokenizer 
import torch
import json
from typing import List, Dict 
import matplotlib.pyplot as plt
import numpy as np


class SentimentAnalyzer:
    """Класс для анализа тональности текстов с использованием трансформера"""
    
    def __init__(self, model_name: str = 'blanchefort/rubert-base-cased-sentiment'):
        print("Инициализация анализатора тональности...")
        
        # Трансформер для анализа тональности
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            trust_remote_code=True, 
            attn_implementation='eager'
        )
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        print(f"Устройство: {self.device}")

    def predict_sentiment(self, texts: List[str], batch_size: int = 8) -> List[float]:
        """
        Предсказывает тональность для списка текстов (с батчингом для скорости)
        
        Args:
            texts: список текстов для анализа
            batch_size: размер батча для обработки
        """
        self.model.eval()
        sentiment_out = []
        
        # Обрабатываем батчами для ускорения
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch_texts, 
                    return_tensors='pt', 
                    truncation=True, 
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                logits = self.model(**inputs).logits
                proba = torch.softmax(logits, dim=1).cpu().numpy()
                
                sentiment_out.extend([float(p[0]) for p in proba])
        
        return sentiment_out

    
class YouTubeSentimentAnalyzer(SentimentAnalyzer):
    """Класс для анализа тональности YouTube комментариев"""
    
    def __init__(self, model_name: str = 'blanchefort/rubert-base-cased-sentiment'):
        super().__init__(model_name)

    def load_comments_from_json(self, json_file: str) -> List[Dict]:
        """Загружает комментарии из JSON файла"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('comments', [])

    def analyze_from_json(self, json_file: str,
                         output_plot: str = 'sentiment_analysis.png',
                         output_pie_chart: str = 'sentiment_pie_chart.png',
                         batch_size: int = 8) -> Dict:
        """
        Анализирует тональность комментариев из JSON
        
        Args:
            json_file: путь к JSON файлу с комментариями
            output_plot: путь для сохранения графика
            output_pie_chart: путь для сохранения круговой диаграммы
            batch_size: размер батча для анализа тональности
            
        Returns:
            Dict: результаты анализа
        """
        print(f"\n{'='*60}")
        print(f"Анализ тональности комментариев")
        print(f"{'='*60}\n")
        
        print(f"Загружаю комментарии из {json_file}...")
        comments = self.load_comments_from_json(json_file)
        
        if not comments:
            print("❌ Нет комментариев для анализа")
            return {}
        
        print(f"✓ Найдено {len(comments)} комментариев")
        
        # Извлекаем тексты
        texts = [c.get('text', '') for c in comments if c.get('text')]
        
        if not texts:
            print("❌ Нет текстов для анализа")
            return {}
        
        print(f"\n📊 Анализ тональности {len(texts)} комментариев...")
        print(f"Модель: {self.model_name}")
        print(f"Batch size: {batch_size}")
        
        sentiments = self.predict_sentiment(texts, batch_size=batch_size)
        print(f"✓ Анализ завершен")
        
        # Создаем график
        self._create_plot(sentiments, output_plot)
        
        # Статистика
        results = self._calculate_statistics(comments, sentiments, output_plot)
        self._print_statistics(results)

        # Создаем круговую диаграмму
        self._create_pie_chart(results, output_pie_chart)
        results['output_pie_chart'] = output_pie_chart
        
        return results

    def _create_pie_chart(self, results: Dict, output_pie_chart: str):
        """Создает и сохраняет круговую диаграмму распределения тональности"""
        labels = ['Позитивные', 'Нейтральные', 'Негативные']
        sizes = [results['positive_percentage'], results['neutral_percentage'], results['negative_percentage']]
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        explode = (0.05, 0, 0)  # "взорвать" первый кусок

        plt.figure(figsize=(8, 8), dpi=150)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.title('Распределение тональности комментариев', fontsize=14, fontweight='bold')
        plt.axis('equal')
        plt.savefig(output_pie_chart)
        plt.close()

        print(f"✓ Круговая диаграмма сохранена: {output_pie_chart}")
    
    def _create_plot(self, sentiments: List[float], output_plot: str):
        """Создает и сохраняет график анализа"""
        plt.figure(figsize=(14, 7), dpi=300)
        
        # Основной график
        plt.subplot(2, 1, 1)
        plt.plot(sentiments, label='Тональность комментариев', color='#3498db', linewidth=1.5, alpha=0.8)
        
        plt.xlabel('Номер комментария', fontsize=11)
        plt.ylabel('Тональность', fontsize=11)
        plt.title(f'Анализ тональности комментариев (всего: {len(sentiments)})', 
                 fontsize=13, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3, linestyle='--')
        
        # Гистограмма распределения
        plt.subplot(2, 1, 2)
        plt.hist(sentiments, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
        plt.xlabel('Тональность', fontsize=11)
        plt.ylabel('Количество комментариев', fontsize=11)
        plt.title('Распределение тональности', fontsize=12, fontweight='bold')
        plt.axvline(np.mean(sentiments), color='#e74c3c', linestyle='--', 
                   linewidth=2, label=f'Среднее: {np.mean(sentiments):.3f}')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3, linestyle='--', axis='y')
        
        plt.tight_layout()
        plt.savefig(output_plot)
        plt.close()
        
        print(f"✓ График сохранен: {output_plot}")
    
    def _calculate_statistics(self, comments: List[Dict], sentiments: List[float],
                             output_plot: str) -> Dict:
        """Вычисляет статистику анализа"""
        positive = sum(1 for s in sentiments if s > 0.6)
        negative = sum(1 for s in sentiments if s < 0.4)
        neutral = len(sentiments) - positive - negative
        
        return {
            "total_comments": len(comments),
            "analyzed_comments": len(sentiments),
            "average_sentiment": float(np.mean(sentiments)),
            "median_sentiment": float(np.median(sentiments)),
            "std_sentiment": float(np.std(sentiments)),
            "min_sentiment": float(np.min(sentiments)),
            "max_sentiment": float(np.max(sentiments)),
            "positive_comments": positive,
            "negative_comments": negative,
            "neutral_comments": neutral,
            "positive_percentage": (positive / len(sentiments)) * 100,
            "negative_percentage": (negative / len(sentiments)) * 100,
            "neutral_percentage": (neutral / len(sentiments)) * 100,
            "output_plot": output_plot
        }
    
    def _print_statistics(self, results: Dict):
        """Выводит статистику в консоль"""
        print(f"\n{'='*60}")
        print(f"📈 РЕЗУЛЬТАТЫ АНАЛИЗА")
        print(f"{'='*60}")
        print(f"Всего комментариев: {results['analyzed_comments']}")
        print(f"\n📊 Статистика тональности:")
        print(f"  • Среднее:  {results['average_sentiment']:.3f}")
        print(f"  • Медиана:  {results['median_sentiment']:.3f}")
        print(f"  • Ст. откл: {results['std_sentiment']:.3f}")
        print(f"  • Мин:      {results['min_sentiment']:.3f}")
        print(f"  • Макс:     {results['max_sentiment']:.3f}")
        print(f"\n😊 Распределение:")
        print(f"  • Позитивные (>0.6): {results['positive_comments']} ({results['positive_percentage']:.1f}%)")
        print(f"  • Нейтральные:       {results['neutral_comments']} ({results['neutral_percentage']:.1f}%)")
        print(f"  • Негативные (<0.4): {results['negative_comments']} ({results['negative_percentage']:.1f}%)")
        print(f"{'='*60}\n")
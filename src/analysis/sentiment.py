from transformers import AutoModelForSequenceClassification, AutoTokenizer 
import torch
import torch.nn as nn
import json
from typing import List, Dict 
import matplotlib.pyplot as plt
import numpy as np

class LSTMSmoother(nn.Module):
    """LSTM модель для сглаживания временного ряда тональности"""
    
    def __init__(self, hidden_size: int = 32, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMSmoother, self).__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, 1)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out)
        # lstm_out shape: (batch, seq_len, hidden_size * 2)
        out = self.fc(lstm_out)
        return out.squeeze(-1)


class SentimentAnalyzer:
    """Гибридный класс: Трансформер для анализа + LSTM для сглаживания"""
    
    def __init__(self, model_name: str = 'Tochka-AI/ruRoPEBert-classic-base-2k'):
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
        
        # LSTM для сглаживания временного ряда
        self.lstm_smoother = None

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
                proba = torch.sigmoid(logits).cpu().numpy()
                
                sentiment_out.extend([float(p[0]) for p in proba])
        
        return sentiment_out
    
    def train_lstm_smoother(self, data: List[float], hidden_size: int = 32, 
                           num_layers: int = 2, epochs: int = 100, 
                           lr: float = 0.01, patience: int = 10) -> LSTMSmoother:
        """
        Обучает LSTM модель для сглаживания временного ряда
        
        Args:
            data: исходные данные тональности
            hidden_size: размер скрытого слоя LSTM
            num_layers: количество слоев LSTM
            epochs: максимальное количество эпох
            lr: learning rate
            patience: количество эпох без улучшения для early stopping
        """
        if len(data) < 10:
            return None
        
        print(f"\nОбучение LSTM сглаживателя...")
        print(f"Архитектура: {num_layers} слоя, hidden_size={hidden_size}")
        
        # Инициализируем модель
        model = LSTMSmoother(hidden_size, num_layers).to(self.device)
        
        # Подготовка данных
        data_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(-1).to(self.device)
        target = data_tensor.clone()
        
        # Оптимизация с weight decay для регуляризации
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Early stopping
        best_loss = float('inf')
        patience_counter = 0
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(data_tensor)
            loss = criterion(output, target.squeeze(-1))
            loss.backward()
            
            # Gradient clipping для стабильности
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step(loss)
            
            # Early stopping
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 20 == 0:
                print(f"Эпоха {epoch + 1}/{epochs}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            if patience_counter >= patience:
                print(f"Early stopping на эпохе {epoch + 1}")
                break
        
        print(f"Обучение завершено. Финальная loss: {best_loss:.6f}")
        return model
    
    def apply_lstm_smoothing(self, data: List[float], lstm_model: LSTMSmoother) -> List[float]:
        """Применяет обученную LSTM модель для сглаживания"""
        if lstm_model is None or len(data) < 10:
            return data
        
        lstm_model.eval()
        data_tensor = torch.FloatTensor(data).unsqueeze(0).unsqueeze(-1).to(self.device)
        
        with torch.no_grad():
            smoothed = lstm_model(data_tensor)
            smoothed = smoothed.cpu().numpy()[0]
        
        return smoothed.tolist() 
    
class YouTubeSentimentAnalyzer(SentimentAnalyzer):
    """Класс для анализа тональности YouTube комментариев"""
    
    def __init__(self, model_name: str = 'Tochka-AI/ruRoPEBert-classic-base-2k'):
        super().__init__(model_name)

    def load_comments_from_json(self, json_file: str) -> List[Dict]:
        """Загружает комментарии из JSON файла"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('comments', [])

    def analyze_from_json(self, json_file: str, 
                         output_plot: str = 'sentiment_analysis.png',
                         use_lstm_smoothing: bool = True,
                         lstm_hidden_size: int = 64,
                         lstm_layers: int = 2,
                         lstm_epochs: int = 100,
                         batch_size: int = 8) -> Dict:
        """
        Анализирует тональность комментариев из JSON
        
        Args:
            json_file: путь к JSON файлу с комментариями
            output_plot: путь для сохранения графика
            use_lstm_smoothing: использовать LSTM сглаживание
            lstm_hidden_size: размер скрытого слоя LSTM
            lstm_layers: количество слоев LSTM
            lstm_epochs: количество эпох обучения LSTM
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
        
        # Применяем LSTM сглаживание
        filtered_sentiments = sentiments
        if use_lstm_smoothing and len(sentiments) > 10:
            lstm_model = self.train_lstm_smoother(
                sentiments, 
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                epochs=lstm_epochs
            )
            filtered_sentiments = self.apply_lstm_smoothing(sentiments, lstm_model)
            print(f"✓ LSTM сглаживание применено")
        
        # Создаем график
        self._create_plot(sentiments, filtered_sentiments, output_plot, use_lstm_smoothing)
        
        # Статистика
        results = self._calculate_statistics(comments, sentiments, filtered_sentiments, output_plot)
        self._print_statistics(results)
        
        return results
    
    def _create_plot(self, sentiments: List[float], filtered: List[float], 
                    output_plot: str, use_smoothing: bool):
        """Создает и сохраняет график анализа"""
        plt.figure(figsize=(14, 7), dpi=300)
        
        # Основной график
        plt.subplot(2, 1, 1)
        plt.plot(sentiments, alpha=0.4, label='Исходные данные', color='#3498db', linewidth=1)
        
        if use_smoothing:
            plt.plot(filtered, linewidth=2.5, label='LSTM сглаживание', color='#e74c3c')
        
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
                             filtered: List[float], output_plot: str) -> Dict:
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
            "smoothed_average": float(np.mean(filtered)),
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
        print(f"\n🎯 Сглаженное среднее: {results['smoothed_average']:.3f}")
        print(f"{'='*60}\n")

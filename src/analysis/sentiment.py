from transformers import AutoTokenizer, AutoModel
import torch 
import matplotlib.pyplot as plt 

class SentimentAnalyzer: 
    def __init__(self):
        self.model_name = 'Tochka-AI/ruRoPEBert-classic-base-2k'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True, attn_implementation='eager')  
        if torch.cuda.is_available():
            self.model = self.model.to('cuda')

    def predict_sentiment(self, text):
        sentiment_out = []
        for sentence in text:
            with torch.no_grad():
                inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True).to(model.device) 
                proba = torch.sigmoid(self.model(**inputs).logits).cpu().numpy()[0] 
                sentiment_out.append(proba)
        return sentiment_out
    
    def ensemble_filter(data: list, n_filters=100, polyorder=0, **savgol_args) -> list:
        filt = 0
        start = len(data)//10
        stop = len(data)//4
        step = (stop-start)//n_filters
        if step == 0:
            step = 1
        for window_size in range(start, stop, step):
            res = savgol_filter(data, window_length=window_size, polyorder=polyorder, **savgol_args)
            filt += res
        return filt/n_filters

class VisualSentimentAnalyzer(SentimentAnalyzer):
    def __init__(self):
        super().__init__()

    def save_sentiment(self, text):
        sentiments = self.predict_sentiment(text)
        filtered_sentiments = savgol_filter(sentiments, window_length=len(sentiments)//15, polyorder=0)
        plt.figure(figsize=(8,6), dpi=300)
        plt.plot(filtered_sentiments)
        plt.xlabel('')
        plt.ylabel('Тональность')
        plt.grid()
        plt.savefig('sentiment_analysis.png') 

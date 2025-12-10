from wordcloud import WordCloud
import matplotlib.pyplot as plt

class WordCloudGenerator:
    def __init__(self, text):
        self.text = text

    def generate_wordcloud(self):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(self.text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.show() 

    def save_wordcloud(self, filename):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(self.text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(filename)
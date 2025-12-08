from wordcloud import WordCloud
import matplotlib.pyplot as plt


def generate_wordcloud(text, output_path):
    """
    Generate a word cloud from the given text and save it to the output path.
    """
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()

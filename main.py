from src.scrapers.youtube_selenium import YouTubeCommentsSaver
from src.analysis.sentiment import YouTubeSentimentAnalyzer
from src.analysis.wordclouds import WordCloudGenerator
import json


if __name__ == "__main__":
    # scraper = YouTubeCommentsSaver(headless=True, slow_mode=True)
    # video_url = "https://www.youtube.com/watch?v=2quS1cbZ0_Q"
    # scraper.save_to_json(
    #     video_url=video_url,
    #     output_file="comments.json",
    #     scroll_pause=2.0,
    #     max_comments=100, 
    #     debug=True
    # )
    
    # analyzer = YouTubeSentimentAnalyzer()
    
    # results = analyzer.analyze_from_json(
    #     json_file='comments.json',
    #     output_plot='sentiment_hybrid_analysis.png',
    #     use_lstm_smoothing=True,
    #     lstm_hidden_size=64,
    #     lstm_layers=2,
    #     lstm_epochs=100,
    #     batch_size=8
    # ) 

    world_cloud = WordCloudGenerator() 
    world_cloud.generate_wordcloud(
        json_file='comments.json',
        output_file='wordcloud.png',
        max_words=100,
        background_color='white',
        width=800,
        height=400
    )

    print("\nАнализ завершен!")
    # print(json.dumps(results, indent=2, ensure_ascii=False))
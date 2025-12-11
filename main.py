from src.scrapers.youtube_selenium import YouTubeCommentsSaver
from src.analysis.sentiment import YouTubeSentimentAnalyzer
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
    
    analyzer = YouTubeSentimentAnalyzer()
    
    results = analyzer.analyze_from_json(
        json_file='comments.json',
        output_plot='sentiment_hybrid_analysis.png',
        use_lstm_smoothing=True,
        lstm_hidden_size=64,
        lstm_layers=2,
        lstm_epochs=100,
        batch_size=8
    )
    print("\nАнализ завершен!")
    print(json.dumps(results, indent=2, ensure_ascii=False))
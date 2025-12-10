from src.scrapers.youtube_selenium import YouTubeCommentsSaver
from src.analysis.youtube_sentiment import YouTubeSentimentAnalyzer
import json


if __name__ == "__main__":
    scraper = YouTubeCommentsSaver(headless=True)
    video_url = "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"
    scraper.save_to_json(
        video_url=video_url,
        output_file="comments.json",
        max_comments=100
    )
    
    analyzer = YouTubeSentimentAnalyzer()
    results = analyzer.analyze_from_json(
        json_file="comments.json",
        output_plot="sentiment_analysis.png",
        use_ensemble=False
    )
    
    print("\nАнализ завершен!")
    print(json.dumps(results, indent=2, ensure_ascii=False))
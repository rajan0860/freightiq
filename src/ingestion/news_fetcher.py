from __future__ import annotations

import os
import requests
from datetime import datetime, timedelta

class NewsFetcher:
    """Fetches recent supply chain news using NewsAPI."""
    
    def __init__(self):
        self.api_key = os.getenv("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2/everything"
        
    def fetch_disruption_news(self, days_back: int = 2) -> list[dict]:
        """Fetch recent news articles related to supply chain disruptions."""
        
        if not self.api_key or self.api_key == "your_newsapi_key_here":
            # Return dummy data if API key is not configured so the app doesn't crash during demo
            return self._get_dummy_news()
            
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        # Query specifically for supply chain issues
        query = "(port OR shipping OR supply chain OR logistics OR cargo) AND (strike OR congestion OR delay OR disruption OR storm OR blockade)"
        
        params = {
            "q": query,
            "from": from_date,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 10,
            "apiKey": self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            for item in data.get("articles", []):
                if item.get("title") and item.get("description"):
                    articles.append({
                        "id": item.get("url", ""),
                        "title": item.get("title"),
                        "description": item.get("description"),
                        "source": item.get("source", {}).get("name", "Unknown"),
                        "date": item.get("publishedAt"),
                        "type": "news"
                    })
            return articles
            
        except requests.RequestException as e:
            print(f"Warning: Failed to fetch from NewsAPI: {e}")
            return self._get_dummy_news()

    def _get_dummy_news(self) -> list[dict]:
        """Provides fallback data when API keys aren't set."""
        return [
            {
                "id": "dummy-news-1",
                "title": "Port of Rotterdam announces 48-hour strike",
                "description": "Dockworkers at Europe's busiest port have walked out over pay disputes. All container loading has paused.",
                "source": "Logistics Weekly",
                "date": datetime.now().isoformat(),
                "type": "news"
            },
            {
                "id": "dummy-news-2",
                "title": "Severe weather impacts Shanghai shipping lanes",
                "description": "A major typhoon is approaching the coast, forcing vessels to drop anchor and await clearance.",
                "source": "Maritime News",
                "date": datetime.now().isoformat(),
                "type": "news"
            }
        ]

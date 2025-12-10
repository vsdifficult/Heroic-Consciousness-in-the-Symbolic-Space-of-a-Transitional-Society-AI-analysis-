import json
import time
from typing import Iterator, Dict, Optional, List
from pathlib import Path
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from transformers import AutoTokenizer, AutoModel
import torch 
import matplotlib.pyplot as plt 
from scipy.signal import savgol_filter

# Для стабильности используйте undetected_chromedriver:
try:
    import undetected_chromedriver as uc
    _USE_UC = True
except Exception:
    _USE_UC = False


class YouTubeCommentsSaver:
    """Класс для парсинга и сохранения комментариев YouTube в JSON"""
    
    def __init__(self, headless: bool = True, driver_path: Optional[str] = None, slow_mode: bool = False):
        self.headless = headless
        self.driver_path = driver_path
        self.slow_mode = slow_mode

    def _create_driver(self):
        if _USE_UC:
            options = uc.ChromeOptions()
            if self.headless:
                options.headless = True
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            driver = uc.Chrome(options=options)
        else:
            options = Options()
            if self.headless:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            driver = webdriver.Chrome(options=options)
        driver.set_window_size(1200, 1000)
        return driver

    def stream_comments(self, video_url: str, max_comments: int = None, 
                       scroll_pause: float = 1.0) -> Iterator[Dict]:
        """Стриминг комментариев: yield каждой найденной нити комментария"""
        driver = self._create_driver()
        driver.get(video_url)

        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.TAG_NAME, "ytd-comments"))
            )
        except Exception:
            driver.quit()
            return

        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        seen_ids = set()
        yielded = 0
        attempts = 0
        
        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(scroll_pause + (0.5 if self.slow_mode else 0.0))
            
            elems = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
            for e in elems:
                try:
                    comment_elem = e.find_element(By.CSS_SELECTOR, "ytd-comment-renderer")
                except:
                    continue
                    
                try:
                    cid = e.get_attribute("id") or comment_elem.get_attribute("id") or None
                    
                    text = ""
                    try:
                        text = comment_elem.find_element(By.CSS_SELECTOR, "#content-text").text
                    except:
                        pass
                        
                    author = ""
                    try:
                        author = comment_elem.find_element(By.CSS_SELECTOR, "#author-text span").text
                    except:
                        pass
                        
                    time_text = ""
                    try:
                        time_text = comment_elem.find_element(
                            By.CSS_SELECTOR, ".published-time-text a"
                        ).text
                    except:
                        pass
                        
                    likes = ""
                    try:
                        likes = comment_elem.find_element(By.CSS_SELECTOR, "#vote-count-middle").text
                    except:
                        pass
                        
                    obj = {
                        "source": "youtube",
                        "video": video_url,
                        "id": cid or f"{video_url}_{yielded}",
                        "author": author,
                        "text": text,
                        "time": time_text,
                        "likes": likes
                    }
                    
                    if obj["id"] not in seen_ids:
                        seen_ids.add(obj["id"])
                        yielded += 1
                        yield obj
                        
                        if max_comments and yielded >= max_comments:
                            driver.quit()
                            return
                except Exception:
                    continue

            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                attempts += 1
                if attempts >= 3:
                    break
            else:
                last_height = new_height
                attempts = 0

        driver.quit()

    def save_to_json(self, video_url: str, output_file: str, 
                     max_comments: int = None, scroll_pause: float = 1.0) -> int:
        """
        Парсит комментарии и сохраняет в JSON файл
        
        Returns:
            int: количество сохраненных комментариев
        """
        comments = []
        
        print(f"Начинаю парсинг комментариев с {video_url}...")
        for comment in self.stream_comments(video_url, max_comments, scroll_pause):
            comments.append(comment)
            if len(comments) % 10 == 0:
                print(f"Собрано комментариев: {len(comments)}")
        
        # Добавляем метаданные
        data = {
            "video_url": video_url,
            "scraped_at": datetime.now().isoformat(),
            "total_comments": len(comments),
            "comments": comments
        }
        
        # Сохраняем в JSON
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"Сохранено {len(comments)} комментариев в {output_file}")
        return len(comments)




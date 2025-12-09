import time
from typing import Iterator, Dict, Optional
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Для стабильности используйте undetected_chromedriver:
try:
    import undetected_chromedriver as uc
    _USE_UC = True
except Exception:
    _USE_UC = False

class YouTubeSeleniumScraper:
    def __init__(self, headless: bool = True, driver_path: Optional[str]=None, slow_mode: bool=False):
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

    def stream_comments(self, video_url: str, max_comments: int = None, scroll_pause: float = 1.0) -> Iterator[Dict]:
        """
        Стриминг комментариев: yield каждой найденной нити комментария как словарь.
        """
        driver = self._create_driver()
        driver.get(video_url)

        # дождаться секции комментариев
        try:
            WebDriverWait(driver, 20).until(EC.presence_of_element_located((By.TAG_NAME, "ytd-comments")))
        except Exception:
            # возможно видео скрывает комментарии
            driver.quit()
            return

        # скроллим вниз, чтобы загрузились комментарии
        last_height = driver.execute_script("return document.documentElement.scrollHeight")
        seen_ids = set()
        yielded = 0
        attempts = 0
        while True:
            driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
            time.sleep(scroll_pause + (0.5 if self.slow_mode else 0.0))
            # собираем элементы комментариев
            elems = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
            for e in elems:
                try:
                    comment_elem = e.find_element(By.CSS_SELECTOR, "ytd-comment-renderer")
                except:
                    continue
                try:
                    cid = e.get_attribute("id") or comment_elem.get_attribute("id") or None
                    # build fields
                    text = ""
                    try:
                        text = comment_elem.find_element(By.CSS_SELECTOR, "#content-text").text
                    except:
                        text = ""
                    author = ""
                    try:
                        author = comment_elem.find_element(By.CSS_SELECTOR, "#author-text span").text
                    except:
                        author = ""
                    time_text = ""
                    try:
                        time_text = comment_elem.find_element(By.CSS_SELECTOR, ".published-time-text a").text
                    except:
                        time_text = ""
                    likes = ""
                    try:
                        likes = comment_elem.find_element(By.CSS_SELECTOR, "#vote-count-middle").text
                    except:
                        likes = ""
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

            # проверяем окончание прокрутки
            new_height = driver.execute_script("return document.documentElement.scrollHeight")
            if new_height == last_height:
                attempts += 1
                if attempts >= 3:
                    break
            else:
                last_height = new_height
                attempts = 0

        driver.quit()

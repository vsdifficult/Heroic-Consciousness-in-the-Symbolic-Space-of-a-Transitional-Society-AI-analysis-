import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


def parse_youtube_comments(url, limit=None):
    """
    Parses YouTube comments from the given URL.
    Returns a list of dictionaries with 'text' key.
    """
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(options=options)
    driver.get(url)

    # Wait for comments to load
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-comment-thread-renderer"))
        )
    except:
        driver.quit()
        return []

    # Scroll to load more comments
    last_height = driver.execute_script("return document.documentElement.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.documentElement.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.documentElement.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
        if limit and len(driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")) >= limit:
            break

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    comments = []

    for comment in soup.find_all("ytd-comment-thread-renderer"):
        text_element = comment.find("yt-formatted-string", {"id": "content-text"})
        if text_element:
            text = text_element.get_text(strip=True)
            if text:
                comments.append({"text": text})
        if limit and len(comments) >= limit:
            break

    driver.quit()
    return comments

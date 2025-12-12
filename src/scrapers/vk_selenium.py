import json
import time
import re
from typing import Iterator, Dict, Optional, List
from pathlib import Path
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

try:
    import undetected_chromedriver as uc
    _USE_UC = True
except Exception:
    _USE_UC = False


class VKCommentsSaver:
    """Класс для парсинга и сохранения комментариев VK в JSON"""
    
    def __init__(self, headless: bool = False, slow_mode: bool = True):
        self.headless = headless
        self.slow_mode = slow_mode

    def _create_driver(self):
        """Создает WebDriver с настройками для обхода защиты"""
        global _USE_UC
        driver = None
        
        # Пробуем undetected-chromedriver
        if _USE_UC:
            for attempt in range(2):
                try:
                    options = uc.ChromeOptions()
                    if self.headless:
                        options.add_argument("--headless=new")
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-blink-features=AutomationControlled")
                    options.add_argument("--disable-dev-shm-usage")
                    options.add_argument("--lang=ru-RU")
                    
                    print(f"Попытка создать undetected_chromedriver ({attempt + 1}/2)...")
                    driver = uc.Chrome(options=options, use_subprocess=True)
                    print("✓ Успешно создан undetected_chromedriver")
                    break
                except Exception as e:
                    print(f"✗ Ошибка undetected_chromedriver: {e}")
                    if attempt < 1:
                        time.sleep(3)
                    else:
                        print("→ Переключаюсь на обычный Selenium WebDriver")
                        driver = None
        
        # Используем обычный Selenium
        if driver is None:
            try:
                print("Создаю обычный Chrome WebDriver...")
                options = Options()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--lang=ru-RU")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")
                options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                
                driver = webdriver.Chrome(options=options)
                print("✓ Успешно создан Chrome WebDriver")
            except Exception as e:
                print(f"✗ Критическая ошибка создания драйвера: {e}")
                raise e
        
        driver.set_window_size(1920, 1080)
        return driver

    def _parse_vk_url(self, url: str) -> Dict[str, str]:
        """Парсит URL VK и извлекает тип контента и ID"""
        # Примеры URL:
        # https://vk.com/wall-123456_789 (пост в группе)
        # https://vk.com/wall123456_789 (пост на стене пользователя)
        # https://vk.com/club123456 (группа)
        # https://vk.com/id123456 (профиль)
        
        patterns = {
            'wall_post': r'vk\.com/wall(-?\d+)_(\d+)',
            'group': r'vk\.com/(club|public)(\d+)',
            'user': r'vk\.com/id(\d+)',
        }
        
        for content_type, pattern in patterns.items():
            match = re.search(pattern, url)
            if match:
                if content_type == 'wall_post':
                    return {
                        'type': 'wall_post',
                        'owner_id': match.group(1),
                        'post_id': match.group(2),
                        'url': url
                    }
                elif content_type == 'group':
                    return {
                        'type': 'group',
                        'group_id': match.group(2),
                        'url': url
                    }
                elif content_type == 'user':
                    return {
                        'type': 'user',
                        'user_id': match.group(1),
                        'url': url
                    }
        
        return {'type': 'unknown', 'url': url}

    def _scroll_and_load_comments(self, driver, max_scrolls: int = 50):
        """Прокручивает страницу и кликает на 'Показать еще' для загрузки комментариев"""
        print("Прокручиваю страницу и загружаю комментарии...")
        
        for scroll in range(max_scrolls):
            # Прокручиваем вниз
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5 if self.slow_mode else 0.8)
            
            # Ищем и кликаем на кнопки "Показать еще" или "Показать предыдущие комментарии"
            try:
                show_more_buttons = driver.find_elements(
                    By.XPATH, 
                    "//a[contains(@class, 'replies_next') or contains(text(), 'Показать') or contains(text(), 'Show')]"
                )
                
                clicked = False
                for btn in show_more_buttons:
                    try:
                        if btn.is_displayed() and btn.is_enabled():
                            driver.execute_script("arguments[0].click();", btn)
                            print(f"  ✓ Кликнул на кнопку 'Показать еще' (прокрутка {scroll + 1})")
                            clicked = True
                            time.sleep(2 if self.slow_mode else 1)
                    except:
                        continue
                
                if not clicked and scroll > 5:
                    print(f"  Кнопок 'Показать еще' не найдено (прокрутка {scroll + 1})")
                    
            except Exception as e:
                if scroll < 3:
                    print(f"  Ошибка при поиске кнопок: {e}")
        
        print("✓ Прокрутка завершена")

    def stream_comments(self, vk_url: str, max_comments: int = None, 
                       scroll_pause: float = 2.0, debug: bool = False) -> Iterator[Dict]:
        """Стриминг комментариев с VK страницы"""
        driver = self._create_driver()
        
        try:
            url_info = self._parse_vk_url(vk_url)
            print(f"\nТип контента: {url_info['type']}")
            print(f"Открываю URL: {vk_url}")
            
            driver.get(vk_url)
            time.sleep(5)
            
            # Прокручиваем до комментариев и загружаем их
            self._scroll_and_load_comments(driver)
            
            # Ждем появления комментариев
            print("\nОжидание загрузки комментариев...")
            try:
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "wall_reply_text"))
                )
                print("✓ Комментарии найдены")
            except TimeoutException:
                print("✗ Комментарии не найдены или отключены")
                try:
                    # Проверяем сообщение об отключенных комментариях
                    disabled = driver.find_elements(By.XPATH, "//*[contains(text(), 'Комментарии отключены')]")
                    if disabled:
                        print("✗ Комментарии отключены для этого поста")
                except:
                    pass
                driver.quit()
                return
            
            time.sleep(2)
            
            # Различные селекторы для комментариев VK
            comment_selectors = [
                ".wall_reply_text",  # Основной класс комментариев
                "[class*='reply']",
                ".reply",
            ]
            
            seen_ids = set()
            yielded = 0
            
            # Ищем комментарии по различным селекторам
            all_comments = []
            for selector in comment_selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        all_comments = elements
                        print(f"✓ Найдено {len(elements)} элементов с селектором '{selector}'")
                        break
                except:
                    continue
            
            if not all_comments:
                print("✗ Не удалось найти комментарии")
                driver.quit()
                return
            
            print(f"\nНачинаю парсинг {len(all_comments)} комментариев...")
            
            for idx, elem in enumerate(all_comments):
                try:
                    # Получаем ID комментария
                    comment_id = elem.get_attribute("id") or f"comment_{idx}"
                    
                    if comment_id in seen_ids:
                        continue
                    
                    # Текст комментария
                    text = ""
                    try:
                        text = elem.text.strip()
                    except:
                        pass
                    
                    if not text or len(text) < 2:
                        continue
                    
                    # Автор комментария
                    author = ""
                    try:
                        # Ищем автора в родительском элементе
                        parent = elem.find_element(By.XPATH, "./ancestor::*[contains(@class, 'reply')]")
                        author_elem = parent.find_element(By.CSS_SELECTOR, ".author, a.wall_signed_by, [class*='author']")
                        author = author_elem.text.strip()
                    except:
                        try:
                            # Альтернативный способ
                            author_elem = elem.find_element(By.XPATH, ".//preceding::a[contains(@class, 'author')][1]")
                            author = author_elem.text.strip()
                        except:
                            author = "Unknown"
                    
                    # Время публикации
                    time_text = ""
                    try:
                        parent = elem.find_element(By.XPATH, "./ancestor::*[contains(@class, 'reply')]")
                        time_elem = parent.find_element(By.CSS_SELECTOR, ".rel_date, .published, [class*='date']")
                        time_text = time_elem.text.strip()
                    except:
                        pass
                    
                    # Лайки
                    likes = "0"
                    try:
                        parent = elem.find_element(By.XPATH, "./ancestor::*[contains(@class, 'reply')]")
                        likes_elem = parent.find_element(By.CSS_SELECTOR, ".like_count, [class*='like']")
                        likes = likes_elem.text.strip() or "0"
                    except:
                        pass
                    
                    comment_data = {
                        "source": "vk",
                        "url": vk_url,
                        "id": comment_id,
                        "author": author,
                        "text": text,
                        "time": time_text,
                        "likes": likes
                    }
                    
                    seen_ids.add(comment_id)
                    yielded += 1
                    
                    if debug and yielded <= 3:
                        print(f"\n--- Комментарий #{yielded} ---")
                        print(f"Автор: {author}")
                        print(f"Текст: {text[:100]}...")
                        print(f"Время: {time_text}")
                        print(f"Лайки: {likes}")
                    
                    yield comment_data
                    
                    if max_comments and yielded >= max_comments:
                        print(f"\n✓ Достигнут лимит: {max_comments} комментариев")
                        driver.quit()
                        return
                        
                except Exception as e:
                    if debug:
                        print(f"Ошибка парсинга комментария #{idx}: {e}")
                    continue
            
            print(f"\n✓ Парсинг завершен. Собрано {yielded} комментариев")
            
        finally:
            try:
                driver.quit()
            except:
                pass

    def save_to_json(self, vk_url: str, output_file: str, 
                     max_comments: int = None, scroll_pause: float = 2.0, 
                     debug: bool = False) -> int:
        """
        Парсит комментарии VK и сохраняет в JSON файл
        
        Args:
            vk_url: URL поста VK (например, https://vk.com/wall-123456_789)
            output_file: Путь к выходному JSON файлу
            max_comments: Максимальное количество комментариев (None = все)
            scroll_pause: Пауза между прокрутками (секунды)
            debug: Включить отладочный вывод
            
        Returns:
            int: количество сохраненных комментариев
        """
        comments = []
        
        print(f"\n{'='*60}")
        print(f"Начинаю парсинг комментариев VK")
        print(f"URL: {vk_url}")
        print(f"{'='*60}\n")
        
        for comment in self.stream_comments(vk_url, max_comments, scroll_pause, debug):
            comments.append(comment)
            if len(comments) % 10 == 0:
                print(f"📝 Собрано комментариев: {len(comments)}")
        
        # Добавляем метаданные
        url_info = self._parse_vk_url(vk_url)
        data = {
            "url": vk_url,
            "content_type": url_info.get('type', 'unknown'),
            "scraped_at": datetime.now().isoformat(),
            "total_comments": len(comments),
            "comments": comments
        }
        
        # Сохраняем в JSON
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n{'='*60}")
        print(f"✓ Сохранено {len(comments)} комментариев в {output_file}")
        print(f"{'='*60}\n")
        
        return len(comments)
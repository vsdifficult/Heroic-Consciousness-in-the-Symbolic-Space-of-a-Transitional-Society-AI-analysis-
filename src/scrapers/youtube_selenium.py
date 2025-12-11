import json
import time
from typing import Iterator, Dict, Optional
from pathlib import Path
from datetime import datetime

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from http.client import IncompleteRead


try:
    import undetected_chromedriver as uc
    _USE_UC = True
except Exception:
    _USE_UC = False


class YouTubeCommentsSaver:
    """Класс для парсинга и сохранения комментариев YouTube в JSON"""
    
    def __init__(self, headless: bool = False, driver_path: Optional[str] = None, slow_mode: bool = True):
        self.headless = headless
        self.driver_path = driver_path
        self.slow_mode = slow_mode

    def _create_driver(self):
        global _USE_UC
        driver = None
        
        # Сначала пробуем undetected-chromedriver
        if _USE_UC:
            for attempt in range(2):
                try:
                    options = uc.ChromeOptions()
                    if self.headless:
                        options.add_argument("--headless=new")
                    options.add_argument("--no-sandbox")
                    options.add_argument("--disable-blink-features=AutomationControlled")
                    options.add_argument("--disable-dev-shm-usage")
                    options.add_argument("--lang=en-US")
                    
                    print(f"Попытка создать undetected_chromedriver ({attempt + 1}/2)...")
                    driver = uc.Chrome(options=options, use_subprocess=True)
                    print("✓ Успешно создан undetected_chromedriver")
                    break
                except (IncompleteRead, Exception) as e:
                    print(f"✗ Ошибка undetected_chromedriver: {e}")
                    if attempt < 1:
                        time.sleep(3)
                    else:
                        print("→ Переключаюсь на обычный Selenium WebDriver")
                        driver = None
        
        # Если undetected-chromedriver не сработал, используем обычный Selenium
        if driver is None:
            try:
                print("Создаю обычный Chrome WebDriver...")
                options = Options()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-blink-features=AutomationControlled")
                options.add_argument("--lang=en-US")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")
                
                # Добавляем user-agent чтобы выглядеть как обычный браузер
                options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
                
                driver = webdriver.Chrome(options=options)
                print("✓ Успешно создан Chrome WebDriver")
            except Exception as e:
                print(f"✗ Критическая ошибка создания драйвера: {e}")
                raise e
        
        driver.set_window_size(1920, 1080)
        return driver

    def _scroll_to_comments(self, driver):
        """Прокручивает страницу до секции комментариев"""
        print("Прокручиваю до секции комментариев...")
        for i in range(5):
            driver.execute_script("window.scrollBy(0, 400);")
            time.sleep(0.3)

    def _debug_print_html(self, driver):
        """Отладочная функция для вывода структуры комментариев"""
        try:
            threads = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
            if threads:
                print(f"\n=== ОТЛАДКА: Структура первого комментария ===")
                html = threads[0].get_attribute('outerHTML')[:1000]
                print(html)
                print("=" * 50)
        except Exception as e:
            print(f"Ошибка отладки: {e}")

    def stream_comments(self, video_url: str, max_comments: int = None, 
                       scroll_pause: float = 2.0, debug: bool = False) -> Iterator[Dict]:
        """Стриминг комментариев: yield каждой найденной нити комментария"""
        driver = self._create_driver()
        
        try:
            print(f"Открываю URL: {video_url}")
            driver.get(video_url)
            time.sleep(4)
            
            # Прокручиваем до комментариев
            self._scroll_to_comments(driver)
            
            # Ждём появления комментариев
            print("Ожидание загрузки секции комментариев...")
            try:
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "ytd-comments"))
                )
                print("✓ Секция ytd-comments найдена.")
                time.sleep(3)
            except TimeoutException:
                print("✗ Секция комментариев не найдена.")
                driver.quit()
                return

            # Проверяем, не отключены ли комментарии
            try:
                disabled_msg = driver.find_element(By.CSS_SELECTOR, "ytd-message-renderer")
                msg_text = disabled_msg.text.lower()
                if "disabled" in msg_text or "отключен" in msg_text or "turned off" in msg_text:
                    print("✗ Комментарии отключены для этого видео.")
                    driver.quit()
                    return
            except NoSuchElementException:
                pass
            
            if debug:
                self._debug_print_html(driver)
            
            last_height = driver.execute_script("return document.documentElement.scrollHeight")
            seen_ids = set()
            yielded = 0
            no_new_comments_count = 0
            scroll_count = 0
            
            # Все возможные селекторы для разных версий YouTube
            selectors_to_try = [
                ("ytd-comment-thread-renderer", "yt-attributed-string#content-text", "yt-formatted-string#author-text", "span#vote-count-middle"),
                ("ytd-comment-thread-renderer", "#content-text", "#author-text span", "#vote-count-middle"),
                ("ytd-comment-thread-renderer", "yt-formatted-string.ytd-comment-renderer", "#author-text", "#vote-count-middle"),
            ]
            
            while True:
                scroll_count += 1
                print(f"\n--- Прокрутка #{scroll_count} ---")
                
                # Прокручиваем
                driver.execute_script(
                    "window.scrollTo({top: document.documentElement.scrollHeight, behavior: 'smooth'});"
                )
                time.sleep(scroll_pause + (1.0 if self.slow_mode else 0.0))
                
                # Ищем комментарии
                elems = driver.find_elements(By.CSS_SELECTOR, "ytd-comment-thread-renderer")
                print(f"Найдено элементов ytd-comment-thread-renderer: {len(elems)}")
                
                if debug and scroll_count == 1 and len(elems) > 0:
                    print("\n=== Проверяю селекторы на первом элементе ===")
                    test_elem = elems[0]
                    for i, (thread_sel, text_sel, author_sel, likes_sel) in enumerate(selectors_to_try):
                        print(f"\nВариант #{i+1}:")
                        try:
                            text_e = test_elem.find_element(By.CSS_SELECTOR, text_sel)
                            print(f"  ✓ Текст найден: {text_sel} -> '{text_e.text[:50]}...'")
                        except:
                            print(f"  ✗ Текст НЕ найден: {text_sel}")
                        try:
                            author_e = test_elem.find_element(By.CSS_SELECTOR, author_sel)
                            print(f"  ✓ Автор найден: {author_sel} -> '{author_e.text}'")
                        except:
                            print(f"  ✗ Автор НЕ найден: {author_sel}")
                    print("=" * 50)
                
                new_in_batch = 0
                for e in elems:
                    comment_data = None
                    
                    # Пробуем разные наборы селекторов
                    for thread_sel, text_sel, author_sel, likes_sel in selectors_to_try:
                        try:
                            # ID комментария
                            cid = e.get_attribute("id")
                            
                            # Текст
                            text = ""
                            try:
                                text_elem = e.find_element(By.CSS_SELECTOR, text_sel)
                                text = text_elem.text.strip()
                            except NoSuchElementException:
                                continue  # Если текст не найден, пробуем следующий набор селекторов
                            
                            if not text:
                                continue
                            
                            # Автор
                            author = ""
                            try:
                                author_elem = e.find_element(By.CSS_SELECTOR, author_sel)
                                author = author_elem.text.strip()
                            except NoSuchElementException:
                                pass
                            
                            # Время публикации
                            time_text = ""
                            time_selectors = [
                                "a.yt-simple-endpoint.style-scope.yt-formatted-string",
                                "yt-formatted-string.published-time-text a",
                                ".published-time-text a",
                                "a#published-time-text",
                            ]
                            for time_sel in time_selectors:
                                try:
                                    time_elem = e.find_element(By.CSS_SELECTOR, time_sel)
                                    time_text = time_elem.text.strip()
                                    if time_text:
                                        break
                                except NoSuchElementException:
                                    continue
                            
                            # Лайки
                            likes = "0"
                            try:
                                likes_elem = e.find_element(By.CSS_SELECTOR, likes_sel)
                                likes_text = likes_elem.text.strip()
                                likes = likes_text if likes_text else "0"
                            except NoSuchElementException:
                                pass
                            
                            # Если мы дошли сюда, значит нашли текст - создаём объект
                            comment_data = {
                                "source": "youtube",
                                "video": video_url,
                                "id": cid or f"comment_{yielded}",
                                "author": author,
                                "text": text,
                                "time": time_text,
                                "likes": likes
                            }
                            break  # Нашли данные, выходим из цикла селекторов
                            
                        except Exception as ex:
                            if debug:
                                print(f"Ошибка с набором селекторов: {ex}")
                            continue
                    
                    # Если нашли данные комментария, добавляем их
                    if comment_data and comment_data["id"] not in seen_ids:
                        seen_ids.add(comment_data["id"])
                        yielded += 1
                        new_in_batch += 1
                        yield comment_data
                        
                        if max_comments and yielded >= max_comments:
                            print(f"✓ Достигнут лимит: {max_comments} комментариев")
                            driver.quit()
                            return
                
                print(f"Новых комментариев в этой прокрутке: {new_in_batch}")
                print(f"Всего собрано: {yielded}")
                
                # Проверяем изменение высоты
                new_height = driver.execute_script("return document.documentElement.scrollHeight")
                
                if new_in_batch == 0:
                    no_new_comments_count += 1
                else:
                    no_new_comments_count = 0
                
                if new_height == last_height or no_new_comments_count >= 3:
                    print(f"\n✓ Парсинг завершён. Собрано {yielded} комментариев.")
                    break
                
                last_height = new_height

        finally:
            try:
                driver.quit()
            except:
                pass

    def save_to_json(self, video_url: str, output_file: str, 
                     max_comments: int = None, scroll_pause: float = 2.0, debug: bool = False) -> int:
        """
        Парсит комментарии и сохраняет в JSON файл
        
        Returns:
            int: количество сохраненных комментариев
        """
        comments = []
        
        print(f"\n{'='*60}")
        print(f"Начинаю парсинг комментариев")
        print(f"URL: {video_url}")
        print(f"{'='*60}\n")
        
        for comment in self.stream_comments(video_url, max_comments, scroll_pause, debug):
            comments.append(comment)
            if len(comments) % 10 == 0:
                print(f"📝 Собрано комментариев: {len(comments)}")
        
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
        
        print(f"\n{'='*60}")
        print(f"✓ Сохранено {len(comments)} комментариев в {output_file}")
        print(f"{'='*60}\n")
        return len(comments)


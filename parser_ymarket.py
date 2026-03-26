import re
import time
import random
import pandas as pd

try:
    import undetected_chromedriver as uc
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    from bs4 import BeautifulSoup
    SELENIUM_AVAILABLE = True
    UC_AVAILABLE = True
except ImportError:
    UC_AVAILABLE = False
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.common.exceptions import TimeoutException, NoSuchElementException
        from webdriver_manager.chrome import ChromeDriverManager
        from bs4 import BeautifulSoup
        SELENIUM_AVAILABLE = True
    except ImportError:
        SELENIUM_AVAILABLE = False


# ── URL-утилиты ───────────────────────────────────────────────────────────────

def normalize_url(url: str) -> tuple:
    url = url.strip()
    clean = re.sub(r"[?#].*$", "", url).rstrip("/")
    clean = re.sub(r"/reviews$", "", clean)

    if "market.yandex.ru" not in clean:
        raise ValueError(
            "Ссылка не с Яндекс.Маркета. "
            "Ожидается ссылка вида: https://market.yandex.ru/..."
        )

    if not re.search(r"/product--|/card/", clean):
        raise ValueError(
            "Не удалось распознать ссылку на товар. "
            "Убедитесь, что вы копируете ссылку со страницы конкретного товара, "
            "а не из поисковой выдачи или каталога."
        )

    return clean, clean + "/reviews"


def extract_product_name(url: str) -> str:
    """Достаёт читаемое название из URL."""
    m = re.search(r"/product--([^/]+)/", url)
    if m:
        return m.group(1).replace("-", " ").title()
    m = re.search(r"/card/([^/]+)/", url)
    if m:
        return m.group(1).replace("-", " ").title()
    return "Товар"


# ── WebDriver ─────────────────────────────────────────────────────────────────

def _make_driver(headless: bool = True):
    if UC_AVAILABLE:
        opts = uc.ChromeOptions()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--lang=ru-RU")
        driver = uc.Chrome(options=opts, headless=headless)
        return driver
    else:
        opts = Options()
        if headless:
            opts.add_argument("--headless=new")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
        opts.add_argument("--window-size=1920,1080")
        opts.add_argument("--lang=ru-RU")
        opts.add_argument("--disable-blink-features=AutomationControlled")
        opts.add_experimental_option("excludeSwitches", ["enable-automation"])
        opts.add_experimental_option("useAutomationExtension", False)
        opts.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=opts)
        driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"}
        )
        return driver


# ── Парсер ────────────────────────────────────────────────────────────────────

class YandexMarketParser:
    _SEL_TITLE        = "h1[data-autotest-id='product-title'], h1.cia-cs"
    _SEL_RATING       = "span[data-autotest-id='product-rating-value']"
    _SEL_REVIEWS_CNT  = "a[data-autotest-id='reviews-count'], span[data-autotest-id='reviews-count']"
    _SEL_REVIEW_ITEM  = "div[data-autotest-id='review']"
    _SEL_REVIEW_TEXT  = "div[data-autotest-id='review-text'] span, span[itemprop='reviewBody']"
    _SEL_REVIEW_STARS = "div[data-autotest-id='review-rating'] span[aria-label]"
    _SEL_REVIEW_DATE  = "span[data-autotest-id='review-date'], time"
    _SEL_REVIEW_PROS  = "div[data-autotest-id='review-pros'] span"
    _SEL_REVIEW_CONS  = "div[data-autotest-id='review-cons'] span"
    _SEL_NEXT_PAGE    = "a[data-autotest-id='pagination-next']"
    _SEL_CAPTCHA      = "div.CheckboxCaptcha, div.AdvancedCaptcha, div[class*='captcha']"

    def __init__(self, headless: bool = True):
        if not SELENIUM_AVAILABLE:
            raise ImportError(
                "Установите зависимости:\n"
                "pip install selenium webdriver-manager beautifulsoup4"
            )
        self.driver = _make_driver(headless)
        self.wait = WebDriverWait(self.driver, 15)

    def close(self):
        if self.driver:
            self.driver.quit()

    # ── Информация о товаре ───────────────────────────────────────────────

    def get_product_info(self, product_url: str) -> dict:
        _, reviews_url = normalize_url(product_url)
        self.driver.get(reviews_url)
        self._check_captcha()

        try:
            self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "h1, div[data-autotest-id='review']"))
            )
        except TimeoutException:
            raise TimeoutError("Страница не загрузилась за 15 сек.")

        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        title = ""
        title_el = soup.select_one(self._SEL_TITLE)
        if title_el:
            title = title_el.get_text(strip=True)
        if not title:
            title = extract_product_name(product_url)

        rating = None
        rating_el = soup.select_one(self._SEL_RATING)
        if rating_el:
            m = re.search(r"[\d.,]+", rating_el.get_text())
            if m:
                rating = float(m.group().replace(",", "."))

        reviews_count = 0
        count_el = soup.select_one(self._SEL_REVIEWS_CNT)
        if count_el:
            m = re.search(r"\d+", count_el.get_text())
            if m:
                reviews_count = int(m.group())

        return {
            "title":         title,
            "rating":        rating,
            "reviews_count": reviews_count,
            "reviews_url":   reviews_url,
        }

    # ── Загрузка отзывов ──────────────────────────────────────────────────

    def fetch_reviews(
        self,
        product_url: str,
        max_reviews: int = 200,
        pause: float = 2.0,
    ) -> pd.DataFrame:
        _, reviews_url = normalize_url(product_url)
        self.driver.get(reviews_url)
        self._check_captcha()
        time.sleep(2)

        all_reviews: list[dict] = []
        seen_texts: set[str] = set()

        while len(all_reviews) < max_reviews:
            self._scroll_to_bottom()
            time.sleep(random.uniform(1.0, 2.5))

            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            items = soup.select(self._SEL_REVIEW_ITEM)
            if not items:
                items = soup.select("div[class*='review']:has(span[itemprop='reviewBody'])")

            new_found = 0
            for item in items:
                text_el = item.select_one(self._SEL_REVIEW_TEXT)
                text = text_el.get_text(" ", strip=True) if text_el else ""

                pros_el = item.select_one(self._SEL_REVIEW_PROS)
                cons_el = item.select_one(self._SEL_REVIEW_CONS)
                pros = pros_el.get_text(" ", strip=True) if pros_el else ""
                cons = cons_el.get_text(" ", strip=True) if cons_el else ""

                full_text = " ".join(filter(None, [text, pros, cons]))
                if not full_text or len(full_text) < 5:
                    continue
                if full_text in seen_texts:
                    continue
                seen_texts.add(full_text)

                date_el = item.select_one(self._SEL_REVIEW_DATE)
                date = date_el.get_text(strip=True) if date_el else ""

                rating = None
                stars_el = item.select_one(self._SEL_REVIEW_STARS)
                if stars_el:
                    aria = stars_el.get("aria-label", "")
                    m = re.search(r"([\d.,]+)\s*из", aria)
                    if m:
                        rating = float(m.group(1).replace(",", "."))

                all_reviews.append({
                    "text":   full_text,
                    "rating": rating,
                    "date":   date,
                    "pros":   pros,
                    "cons":   cons,
                })
                new_found += 1

            if new_found == 0:
                next_btn = self._find_next_page()
                if not next_btn:
                    break
                try:
                    self.driver.execute_script("arguments[0].click();", next_btn)
                    time.sleep(random.uniform(pause, pause + 1.5))
                except Exception:
                    break

        return pd.DataFrame(all_reviews[:max_reviews])

    def _check_captcha(self, wait: float = 2.5):
        time.sleep(wait)
        if self.driver.find_elements(By.CSS_SELECTOR, self._SEL_CAPTCHA):
            raise RuntimeError(
                "Яндекс показал капчу. Подождите несколько минут и попробуйте снова."
            )

    def _scroll_to_bottom(self, steps: int = 6):
        for _ in range(steps):
            self.driver.execute_script(
                "window.scrollBy(0, document.body.scrollHeight / 6)"
            )
            time.sleep(random.uniform(0.2, 0.6))

    def _find_next_page(self):
        try:
            return self.driver.find_element(By.CSS_SELECTOR, self._SEL_NEXT_PAGE)
        except NoSuchElementException:
            return None

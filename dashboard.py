import inspect
if not hasattr(inspect, 'getargspec'):
    def getargspec(func):
        spec = inspect.getfullargspec(func)
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)
    inspect.getargspec = getargspec

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformers import pipeline
import re
import os
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="Анализ отзывов",
    layout="wide",
    page_icon="📊",
    menu_items={"About": "Анализ тональности отзывов v1.2 — Жарский Егор"}
)

# ─── Словарь тем ─────────────────────────────────────────────────────────────

@st.cache_data
def load_themes():
    themes_path = os.path.join(os.path.dirname(__file__), "themes.json")
    if os.path.exists(themes_path):
        with open(themes_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "Доставка": ["доставка", "курьер", "срок", "задержка"],
        "Качество":  ["качество", "брак", "дефект", "материал"],
        "Цена":      ["цена", "стоимость", "дорого", "дешево"],
        "Сервис":    ["сервис", "поддержка", "консультация"],
        "Упаковка":  ["упаковка", "повреждение", "коробка"],
    }

THEMES = load_themes()

# ─── Лемматизация ────────────────────────────────────────────────────────────

try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
except ImportError:
    morph = None

def lemmatize_text(text: str) -> str:
    text = text.lower()
    if morph:
        words = re.findall(r"\w+", text)
        return " ".join(morph.parse(w)[0].normal_form for w in words)
    return text

# ─── Модель ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Загружаю модель RuBERT...")
def load_model():
    return pipeline(
        task="sentiment-analysis",
        model="blanchefort/rubert-base-cased-sentiment",
        device=-1,
    )

# ─── CSV ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        return None

# ─── Темы ────────────────────────────────────────────────────────────────────

def detect_themes(text: str) -> list:
    processed = lemmatize_text(text)
    found = []
    for theme, keywords in THEMES.items():
        for kw in keywords:
            kw_norm = morph.parse(kw)[0].normal_form if morph else kw.lower()
            if re.search(rf"\b{re.escape(kw_norm)}\b", processed):
                found.append(theme)
                break
    return found if found else ["Другое"]

def calculate_stats(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for theme in list(THEMES.keys()) + ["Другое"]:
        subset = data[data["themes"].apply(lambda x: theme in x)]
        total = len(subset)
        if total == 0:
            continue
        pos = (subset["sentiment"] == "POSITIVE").sum()
        neg = (subset["sentiment"] == "NEGATIVE").sum()
        neu = (subset["sentiment"] == "NEUTRAL").sum()
        rows.append({
            "Тема": theme, "Всего": total,
            "Позитивные":  round(pos / total * 100, 1),
            "Негативные":  round(neg / total * 100, 1),
            "Нейтральные": round(neu / total * 100, 1),
        })
    return pd.DataFrame(rows)

# ─── Анализ ──────────────────────────────────────────────────────────────────

def run_analysis(data: pd.DataFrame) -> pd.DataFrame:
    texts = data["text"].astype(str).apply(lambda x: x[:1024]).tolist()
    model = load_model()
    progress = st.progress(0, text="Анализирую тональность...")
    batch_size, predictions = 32, []
    for i in range(0, len(texts), batch_size):
        try:
            predictions.extend(model(texts[i: i + batch_size]))
        except Exception as e:
            st.error(f"Ошибка модели: {e}")
            return data
        pct = min(int((i + batch_size) / len(texts) * 100), 100)
        progress.progress(pct, text=f"Анализирую тональность... {pct}%")
    progress.empty()
    data = data.copy()
    data["sentiment"] = [p["label"] for p in predictions]
    with st.spinner("Определяю темы..."):
        data["themes"] = data["text"].astype(str).apply(detect_themes)
    return data

# ─── Отображение результатов ─────────────────────────────────────────────────

COLOR_MAP = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#f1c40f"}

def show_results(data: pd.DataFrame):
    stats = calculate_stats(data)
    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Позитивные", f"{round((data['sentiment']=='POSITIVE').mean()*100,1)}%")
    c2.metric("❌ Негативные", f"{round((data['sentiment']=='NEGATIVE').mean()*100,1)}%")
    c3.metric("➖ Нейтральные", f"{round((data['sentiment']=='NEUTRAL').mean()*100,1)}%")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Распределение тональности**")
        counts = data["sentiment"].value_counts().reset_index()
        counts.columns = ["sentiment", "count"]
        fig = px.pie(counts, names="sentiment", values="count",
                     color="sentiment", color_discrete_map=COLOR_MAP, hole=0.35)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10,b=10,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("**Тональность по темам**")
        df = stats[stats["Тема"] != "Другое"].sort_values("Всего", ascending=True)
        fig2 = go.Figure()
        for col, color in [("Позитивные","#2ecc71"),("Нейтральные","#f1c40f"),("Негативные","#e74c3c")]:
            fig2.add_trace(go.Bar(name=col, y=df["Тема"], x=df[col],
                                  orientation="h", marker_color=color))
        fig2.update_layout(barmode="stack", xaxis_title="% отзывов",
                           margin=dict(t=10,b=10,l=10,r=10),
                           legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("**Статистика по темам**")
    st.dataframe(stats.sort_values("Всего", ascending=False),
                 use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("**Просмотр отзывов**")
    cf1, cf2 = st.columns(2)
    with cf1:
        sel_theme = st.selectbox("Тема:", sorted(data["themes"].explode().unique()),
                                 key="theme_sel")
    with cf2:
        sel_sent = st.selectbox("Тональность:", ["Все","POSITIVE","NEGATIVE","NEUTRAL"],
                                key="sent_sel")
    filtered = data[data["themes"].apply(lambda x: sel_theme in x)]
    if sel_sent != "Все":
        filtered = filtered[filtered["sentiment"] == sel_sent]
    st.caption(f"Найдено: {len(filtered)} отзывов")
    show_cols = ["text","sentiment"] + (["rating"] if "rating" in filtered.columns else [])
    st.dataframe(
        filtered[show_cols].rename(columns={"text":"Отзыв","sentiment":"Тональность","rating":"Оценка"}),
        use_container_width=True, hide_index=True, height=280
    )

    st.markdown("---")
    csv_out = data[["text","sentiment","themes"]].copy()
    csv_out["themes"] = csv_out["themes"].apply(lambda x: ", ".join(x))
    st.download_button("⬇ Скачать результаты (CSV)",
                       csv_out.to_csv(index=False, encoding="utf-8-sig"),
                       file_name="analysis_results.csv", mime="text/csv")

# ─── Вкладка CSV ─────────────────────────────────────────────────────────────

def tab_csv():
    st.subheader("Загрузка CSV-файла")
    uploaded = st.file_uploader("CSV с отзывами (обязательный столбец **text**)", type=["csv"])
    if uploaded is None:
        with st.expander("Пример формата файла"):
            st.code("text\nОтличный товар, доставили быстро!\nКачество плохое, брак.\nНормально, всё ок.")
        return
    data = load_csv(uploaded)
    if data is None:
        return
    if "text" not in data.columns:
        st.error(f"Нет столбца **text**. Найдены столбцы: {list(data.columns)}")
        return
    max_rows = st.sidebar.number_input("Макс. отзывов", 100, 50000, 5000, 500)
    if len(data) > max_rows:
        data = data.sample(max_rows, random_state=42).reset_index(drop=True)
        st.info(f"Выборка ограничена до {max_rows} отзывов.")
    st.success(f"Загружено {len(data):,} отзывов.")
    if st.button("▶ Запустить анализ", type="primary", key="btn_csv"):
        st.session_state["csv_results"] = run_analysis(data)
    if "csv_results" in st.session_state:
        show_results(st.session_state["csv_results"])

# ─── Вкладка Яндекс.Маркет ───────────────────────────────────────────────────

def tab_ymarket():
    st.subheader("Парсинг отзывов с Яндекс.Маркета по ссылке")

    with st.expander("📖 Как получить ссылку на товар"):
        st.markdown(
            "1. Откройте товар на [market.yandex.ru](https://market.yandex.ru)\n"
            "2. Скопируйте URL из адресной строки браузера\n"
            "3. Вставьте в поле ниже\n\n"
            "**Пример ссылки:**\n"
            "`https://market.yandex.ru/product--smartfon-xiaomi-redmi-note-13/1926271090`"
        )

    with st.form("ym_url_form"):
        url_input = st.text_input(
            "Ссылка на товар",
            placeholder="https://market.yandex.ru/product--название-товара/ID",
        )
        max_reviews = st.slider("Максимум отзывов", 20, 300, 100, 10)
        submitted = st.form_submit_button("▶ Загрузить и анализировать", type="primary")

    if not submitted or not url_input.strip():
        st.info("Вставьте ссылку на товар и нажмите «Загрузить и анализировать».")
        return

    try:
        from parser_ymarket import normalize_url, extract_product_name
        product_url, reviews_url = normalize_url(url_input.strip())
    except ValueError as e:
        st.error(str(e))
        return

    product_name = extract_product_name(product_url)
    st.markdown(f"**Товар:** {product_name}")
    st.markdown(f"🔗 [Страница отзывов]({reviews_url})")

    parser = None
    try:
        with st.spinner("Подключаюсь к Яндекс.Маркету..."):
            parser = YandexMarketParser(headless=True)

        with st.spinner("Читаю информацию о товаре..."):
            info = parser.get_product_info(product_url)

        # Показываем карточку товара
        col1, col2, col3 = st.columns(3)
        col1.metric("Название", info["title"][:40] + ("…" if len(info["title"]) > 40 else ""))
        col2.metric("Рейтинг", f"★ {info['rating']}" if info["rating"] else "—")
        col3.metric("Отзывов на сайте", f"{info['reviews_count']:,}" if info["reviews_count"] else "—")

        with st.spinner(f"Загружаю до {max_reviews} отзывов... (~1–2 мин)"):
            df = parser.fetch_reviews(product_url, max_reviews=max_reviews)

    except RuntimeError as e:
        st.error(str(e))
        return
    except TimeoutError as e:
        st.error(f"Таймаут: {e}")
        return
    except Exception as e:
        st.error(f"Ошибка: {e}")
        return
    finally:
        if parser:
            parser.close()

    if df.empty:
        st.warning(
            "Отзывы не загрузились. Возможные причины:\n"
            "- Яндекс заблокировал запрос (подождите 5–10 минут)\n"
            "- У товара нет отзывов"
        )
        return

    st.success(f"Загружено {len(df)} отзывов.")

    with st.spinner("Анализирую тональность..."):
        df = run_analysis(df)

    st.session_state["ym_results"] = df

    if "ym_results" in st.session_state:
        show_results(st.session_state["ym_results"])


def main():
    st.title("📊 Анализ тональности отзывов")
    st.caption("Автоматическая классификация по тональности и темам")

    with st.sidebar:
        st.header("Настройки")
        st.markdown("**Темы из themes.json:**")
        for theme in THEMES:
            st.markdown(f"• {theme} ({len(THEMES[theme])} слов)")

    tab1, tab2 = st.tabs(["📂 CSV-файл", "🌐 Яндекс.Маркет"])
    with tab1:
        tab_csv()
    with tab2:
        tab_ymarket()

if __name__ == "__main__":
    main()

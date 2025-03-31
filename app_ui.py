# app_ui.py
import streamlit as st
import yaml
import os
import pandas as pd
import plotly.express as px
import psutil
import gc

from src.data.data_processing import load_data, show_dataset_stats



CONFIG_PATH = "config/config.yaml"

def load_config(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл конфигурации {path} не найден.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    metrics_dict = data.get("metrics_dict", {})
    ag_models = data.get("ag_models", {})
    return metrics_dict, ag_models

METRICS_DICT, AG_MODELS = load_config(CONFIG_PATH)
metrics_list = list(METRICS_DICT.keys())
presets_list = ["fast_training", "medium_quality", "high_quality", "best_quality"]
all_models_opt = "* (все)"
model_keys = list(AG_MODELS.keys())
model_choices = [all_models_opt] + model_keys

def setup_ui():
    # Стилизуем кнопки: красная для обучения, голубые для остальных
    button_css = """
    <style>
    /* Общий стиль для всех кнопок - одинаковая ширина */
    div[data-testid="stButton"] button {
        width: 100%;
    }

    /* Стиль для кнопки обучения модели (красная с белым текстом) */
    div[data-testid="stButton"] button[kind="secondary"][data-testid="StyledFullScreenButton"] {
        background-color: #E53935;
        color: white;
    }

    /* Стиль для всех остальных кнопок (голубые) */
    div[data-testid="stButton"] button[kind="secondary"]:not([data-testid="StyledFullScreenButton"]) {
        background-color: #2196F3;
        color: white;
    }
    </style>
    """
    st.sidebar.markdown(button_css, unsafe_allow_html=True)

    st.markdown("### Версия 3.0")
    st.title("Бизнес-приложение для прогнозирования временных рядов")
    
    # Добавляем страницу анализа данных
    pages = ["Главная", "Анализ данных", "Help"]
    page_choice = st.sidebar.selectbox("Навигация", pages, key="page_choice")
    
    # Инициализация session_state
    session_keys = [
        "df", "predictor", "leaderboard", "predictions", "fit_summary",
        "static_df_train", "static_df_fore", "best_model_name", "best_model_score",
        "df_forecast", "metric_key", "presets_key", "models_key"
    ]
    for key in session_keys:
        if key not in st.session_state:
            if key == "metric_key":
                st.session_state[key] = metrics_list[0]
            elif key == "presets_key":
                st.session_state[key] = "high_quality"
            elif key == "models_key":
                st.session_state[key] = [all_models_opt]
            else:
                st.session_state[key] = None
    
    if "df_forecast" in st.session_state:
        del st.session_state["df_forecast"]
    if "static_df_fore" in st.session_state:
        del st.session_state["static_df_fore"]
    
    # Если выбрана страница анализа данных или справки, не отображаем остальные элементы UI
    if page_choice in ["Анализ данных", "Help"]:
        return page_choice
    
    # ========== (1) Загрузка данных ==========
    st.sidebar.header("1. Загрузка данных")
    
    # Опциональные настройки загрузки для больших файлов
    with st.sidebar.expander("Настройки для больших файлов"):
        chunk_size = st.number_input(
            "Размер чанка (строк)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=10000,
            help="Для больших файлов (>100 МБ) данные будут загружаться частями. Задайте размер каждой части."
        )
    
    train_file = st.sidebar.file_uploader("Train (обязательно)", type=["csv", "xls", "xlsx"], key="train_file_uploader")
    if st.sidebar.button("📂 Загрузить данные", key="load_data_btn"):
        if not train_file:
            st.error("Train-файл обязателен!")
        else:
            try:
                with st.spinner("Загрузка данных..."):
                    df_train = load_data(train_file, chunk_size=chunk_size)
                    st.session_state["df"] = df_train
                    st.success(f"Train-файл загружен! Строк: {len(df_train)}, колонок: {len(df_train.columns)}")
                    
                    # Для больших датафреймов используем выборку при отображении
                    if len(df_train) > 1000:
                        st.dataframe(df_train.head(1000))
                        st.info(f"Показаны первые 1000 из {len(df_train)} строк.")
                    else:
                        st.dataframe(df_train)
                    
                    st.subheader("Статистика Train")
                    show_dataset_stats(df_train)
                    
                    # Освобождаем память, если её мало
                    memory_usage = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024 * 1024)  # в ГБ
                    if memory_usage > 1.5:  # если используется больше 1.5 ГБ
                        gc.collect()
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")
    
    # ========== (2) Настройка колонок ==========
    st.sidebar.header("2. Колонки датасета")
    df_current = st.session_state.get("df")
    all_cols = list(df_current.columns) if df_current is not None else []
    
    if "dt_col_key" not in st.session_state:
        st.session_state["dt_col_key"] = "<нет>"
    if "tgt_col_key" not in st.session_state:
        st.session_state["tgt_col_key"] = "<нет>"
    if "id_col_key" not in st.session_state:
        st.session_state["id_col_key"] = "<нет>"
    
    dt_stored = st.session_state["dt_col_key"]
    tgt_stored = st.session_state["tgt_col_key"]
    id_stored = st.session_state["id_col_key"]
    
    if dt_stored not in ["<нет>"] + all_cols:
        st.session_state["dt_col_key"] = "<нет>"
    if tgt_stored not in ["<нет>"] + all_cols:
        st.session_state["tgt_col_key"] = "<нет>"
    if id_stored not in ["<нет>"] + all_cols:
        st.session_state["id_col_key"] = "<нет>"
    
    dt_col = st.sidebar.selectbox("Колонка с датой", ["<нет>"] + all_cols, key="dt_col_key")
    tgt_col = st.sidebar.selectbox("Колонка target", ["<нет>"] + all_cols, key="tgt_col_key")
    id_col = st.sidebar.selectbox("Колонка ID (категориальный)", ["<нет>"] + all_cols, key="id_col_key")
    
    # Статические признаки
    st.sidebar.header("Статические признаки (до 3)")
    if "static_feats_key" not in st.session_state:
        st.session_state["static_feats_key"] = []
    existing_static_feats = st.session_state["static_feats_key"]
    possible_static = [c for c in all_cols if c not in [dt_col, tgt_col, id_col, "<нет>"]]
    filtered_feats = [feat for feat in existing_static_feats if feat in possible_static]
    if len(filtered_feats) != len(existing_static_feats):
        st.session_state["static_feats_key"] = filtered_feats
    
    static_feats = st.sidebar.multiselect("Статические колонки:", possible_static, key="static_feats_key")
    
    if "use_holidays_key" not in st.session_state:
        st.session_state["use_holidays_key"] = False
    st.sidebar.checkbox("Учитывать праздники РФ?", value=st.session_state["use_holidays_key"], key="use_holidays_key")
    
    # ========== (3) Обработка пропусков ==========
    st.sidebar.header("3. Обработка пропусков")
    # Заменить существующий selectbox для fill_method более информативным
    fill_options = [
        "None (оставить как есть)", 
        "Constant=0 (заменить на нули)", 
        "Group mean (среднее по группе)", 
        "Forward fill (протянуть значения)", 
        "Interpolate (линейная интерполяция)", 
        "KNN imputer (k ближайших соседей)"
    ]
    fill_map = {opt: opt.split(" ")[0] for opt in fill_options}
    selected_fill = st.sidebar.selectbox(
        "Способ заполнения пропусков", 
        fill_options, 
        key="fill_method_display"
    )
    st.session_state["fill_method_key"] = fill_map[selected_fill]
    
    if "group_cols_for_fill_key" not in st.session_state:
        st.session_state["group_cols_for_fill_key"] = []
    st.sidebar.multiselect("Колонки для группировки", static_feats, key="group_cols_for_fill_key")
    
    # ========== (4) Частота (freq) ==========
    st.sidebar.header("4. Частота (freq)")
    freq_options = ["auto (угадать)", "D (день)", "H (час)", "M (месяц)", "B (рабочие дни)", "W (неделя)", "Q (квартал)"]
    if "freq_key" not in st.session_state:
        st.session_state["freq_key"] = "auto (угадать)"
    st.sidebar.selectbox("freq", freq_options, index=0, key="freq_key")
    
    # ========== (5) Метрика и модели ==========
    st.sidebar.header("5. Метрика и модели")
    st.sidebar.selectbox("Метрика", metrics_list, index=metrics_list.index(st.session_state["metric_key"]), key="metric_key")
    st.sidebar.multiselect("Модели AutoGluon", model_choices, key="models_key")
    st.sidebar.selectbox("Presets", presets_list, index=presets_list.index(st.session_state["presets_key"]), key="presets_key")
    st.sidebar.number_input(
        "prediction_length", 
        1, 365, 3, 
        key="prediction_length_key",
        help="Горизонт прогноза (количество точек в будущем). Модель будет обучена предсказывать именно на это количество шагов вперед."
    )
    
    st.sidebar.number_input(
        "time_limit (sec)", 
        10, 36000, None, 
        key="time_limit_key",
        help="Максимальное время (в секундах) для обучения всех моделей. Чем больше времени, тем лучше результат, но медленнее обучение.",
        placeholder="Полное выполнение"
    )
    
    st.sidebar.checkbox(
        "Прогнозировать только среднее (mean)?", 
        value=st.session_state.get("mean_only_key", False), 
        key="mean_only_key",
        help="Если включено, модель будет предсказывать только среднее значение (0.5 квантиль). Если выключено, будут предсказаны все квантили (0.1, 0.5, 0.9), что даст интервалы неопределенности."
    )
    
    # Предварительный график
    if df_current is not None and dt_col != "<нет>" and tgt_col != "<нет>":
        try:
            with st.spinner("Построение графика..."):
                df_plot = df_current.copy()
                
                # Для больших датафреймов используем выборку при построении графика
                if len(df_plot) > 10000:
                    st.info(f"Для графика используется выборка из 10000 точек (из {len(df_plot)} строк).")
                    # Если есть ID, берем по несколько ID
                    if id_col != "<нет>":
                        ids = df_plot[id_col].unique()
                        if len(ids) > 10:
                            selected_ids = ids[:10]  # Берем первые 10 ID
                            df_plot = df_plot[df_plot[id_col].isin(selected_ids)]
                        
                        # Если все еще слишком много точек
                        if len(df_plot) > 10000:
                            df_plot = df_plot.sample(10000, random_state=42)
                    else:
                        df_plot = df_plot.sample(10000, random_state=42)
                
                df_plot[dt_col] = pd.to_datetime(df_plot[dt_col], errors="coerce", dayfirst=True)
                df_plot = df_plot.dropna(subset=[dt_col])
                
                if id_col != "<нет>":
                    fig_target = px.line(df_plot.sort_values(dt_col), x=dt_col, y=tgt_col, color=id_col, title="График Target по ID")
                else:
                    fig_target = px.line(df_plot.sort_values(dt_col), x=dt_col, y=tgt_col, title="График Target (без ID)")
                
                fig_target.update_layout(height=400)
                st.subheader("Предварительный анализ Target")
                st.plotly_chart(fig_target, use_container_width=True)
                
                # Освобождаем память
                del df_plot
                gc.collect()
        except Exception as e:
            st.warning(f"Не удалось построить график: {e}")
    
    # ========== (6) Обучение модели ==========
    st.sidebar.header("6. Обучение модели")
    st.sidebar.checkbox("Обучение, Прогноз и Сохранение", key="train_predict_save_checkbox")
    st.sidebar.button("🚀 Обучить модель", key="fit_model_btn", type="primary")
    
    # ========== (7) Прогноз ==========
    st.sidebar.header("7. Прогноз")
    st.sidebar.button("📊 Сделать прогноз", key="predict_btn")
    
    # ========== (8) Сохранение результатов ==========
    st.sidebar.header("8. Сохранение результатов прогноза")
    st.sidebar.button("💾 Сохранить результаты в CSV", key="save_csv_btn")
    st.sidebar.button("💾 Сохранить результаты в Excel", key="save_excel_btn")
    
    # ========== (9) Логи приложения ==========
    st.sidebar.header("9. Логи приложения")
    st.sidebar.button("👁️ Показать логи", key="show_logs_btn")
    st.sidebar.button("📥 Скачать логи", key="download_logs_btn")
    
    # ========== (10) Выгрузка моделей и логов ==========
    st.sidebar.header("10. Выгрузка моделей и логов")
    st.sidebar.button("📦 Скачать архив (модели + логи)", key="download_model_and_logs")
    
    return page_choice
# main.py
import json
import streamlit as st
from llm_interface import parse_task_description, generate_pipeline_code
from pipeline_executor import run_pipeline, collect_metrics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AutoML LLM App", layout="wide")
st.title("🤖 AutoML Pipeline Generator with LLM")

# --- Input from user ---
user_prompt = st.text_area("Опишіть вашу задачу (наприклад, 'класифікація відгуків клієнтів'):")
uploaded_file = st.file_uploader("Завантажте CSV-файл з даними:", type=["csv"])
time_limit = st.number_input("Максимальний час на AutoML (сек):", min_value=60, value=300, step=60)

if st.button("🔧 Згенерувати та запустити пайплайн") and user_prompt and uploaded_file:
    # --- Step 1: Parse prompt to JSON schema ---
    task_schema = parse_task_description(user_prompt)
    st.subheader("🔍 Інтерпретована структура задачі")
    st.json(task_schema)

    # --- Step 2: Generate AutoML code ---
    pipeline_code = generate_pipeline_code(task_schema)
    st.subheader("💻 Згенерований код пайплайну")
    st.code(pipeline_code, language="python")

    # --- Step 3: Save uploaded CSV ---
    with open("input_data.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    df = pd.read_csv("input_data.csv")

    # --- Step 3.0: EDA ---
    st.subheader("📄 Описова статистика")
    st.write("### Типи змінних")
    st.write(df.dtypes)

    st.write("### Основні статистики")
    st.dataframe(df.describe(include='all'))

    st.write("### Кількість пропущених значень")
    st.dataframe(df.isnull().sum())

    # --- Step 3.0.1: Target distribution ---
    st.subheader("🎯 Розподіл цільової змінної")
    if task_schema["target"] in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[task_schema["target"]], kde=True)
        st.pyplot(plt.gcf())
    else:
        st.warning(f"Цільова змінна '{task_schema['target']}' не знайдена у CSV")

    # --- Step 3.0.2: Кореляції та частоти ---
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 1:
        st.subheader("🔗 Кореляції між числовими змінними")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt.gcf())

    categorical_cols = df.select_dtypes(include='object').columns
    for col in categorical_cols:
        st.subheader(f"📦 Частоти для '{col}'")
        st.bar_chart(df[col].value_counts())

    # --- Step 3.1: Split into train/test ---
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv("input_data.csv", index=False)     # для AutoML
    test_df.to_csv("test_data.csv", index=False)       # для метрик

    # --- Step 4: Run AutoML pipeline ---
    st.subheader("🚀 Навчання моделі...")
    model, leaderboard = run_pipeline(pipeline_code, "input_data.csv", time_limit)
    st.write("## 📊 Результати AutoML")
    st.dataframe(leaderboard)

    # --- Step 4.1: Verbose model summary ---
    st.subheader("📌 Найкраща модель")
    best_model = leaderboard.iloc[0]
    st.write(f"🔹 Модель: `{best_model['model']}`")
    st.write(f"🔹 Score: `{best_model['score_val']:.4f}`")
    st.write(f"🔹 Час навчання: `{best_model['fit_time']:.2f}` сек")

    if task_schema["task_type"] == "classification":
        st.info("➡️ Чим вищий accuracy або F1, тим краще.")
    else:
        st.info("➡️ Чим нижчий RMSE / MAE, тим краще.")

    # --- Step 4.2: Feature importance (якщо можливо) ---
    st.subheader("📈 Важливість фіч (Feature Importance)")
    try:
        test_data = pd.read_csv("test_data.csv")
        fi_df = model.feature_importance(test_data)
        st.dataframe(fi_df.head(10))

        plt.figure(figsize=(10, 6))
        sns.barplot(data=fi_df.head(10), x="importance", y="feature")
        plt.title("Top-10 важливих фіч")
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning(f"⚠️ Не вдалося отримати важливість фіч: {e}")

    # --- Step 5: Collect metrics ---
    st.subheader("📉 Метрики на тестових даних")
    metrics, figs = collect_metrics(model, test_data)
    st.json(metrics)

    for fig in figs:
        st.pyplot(fig)

    st.success("✅ Пайплайн успішно завершено!")

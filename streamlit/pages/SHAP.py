# Сервер и система
import streamlit as st
from streamlit.web.cli import main
import os
import sys
import pickle

# Визуализация
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px
import shap
from streamlit_shap import st_shap
import numpy as np

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Пути
ROOT = os.getcwd()
TRAIN_DATASET = os.path.join(ROOT, 'data/train_AIC.csv')
BALANCED_DATASET = os.path.join(ROOT, 'data/balanced_train.csv')
TEST_DATASET = os.path.join(ROOT, 'data/test_AIC.csv')
SUBMISSION_PATH = os.path.join(ROOT, 'submissions/')
MODEL_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_model.dat')
PREC_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_preprocessor.dat')
SHAP_SAVE_PATH = os.path.join(ROOT, 'output/shap_values.dat')
EXPLANATION_SAVE_PATH = os.path.join(ROOT, 'output/explanation.dat')
EXPLAINER_SAVE_PATH = os.path.join(ROOT, 'output/explainer.dat')

# Названия фич для SHAP
FEATURE_NAMES = ['Поставщик 1', 'Поставщик 2', 'Поставщик 3', 'Поставщик 4', 'Поставщик 5', 'Поставщик 6', 'Поставщик 7', 
                 'Поставщик 8', 'Поставщик 9', 'Поставщик 10', 'Поставщик 11', 'Поставщик 12', 'Операционный менеджер 1', 
                 'Операционный менеджер 2', 'Операционный менеджер 3', 'Операционный менеджер 4', 'Операционный менеджер 5', 
                 'Операционный менеджер 6', 'Завод', 'Закупочная организация 1', 'Закупочная организация 2', 
                 'Закупочная организация 3', 'Закупочная организация 4', 'Закупочная организация 5', 
                 'Группа закупок 1', 'Группа закупок 2', 'Группа закупок 3', 'Группа закупок 4', 
                 'Группа закупок 5', 'Группа закупок 6', 'Группа закупок 7', 'Группа закупок 8', 
                 'Группа закупок 9', 'Балансовая единица 1', 'Балансовая единица 2', 'Балансовая единица 3', 'Балансовая единица 4', 
                 'Балансовая единица 5', 'ЕИ 1', 'ЕИ 2', 'ЕИ 3', 'ЕИ 4', 'ЕИ 5', 'Группа материалов 1', 'Группа материалов 2', 
                 'Группа материалов 3', 'Группа материалов 4', 'Группа материалов 5', 'Группа материалов 6', 'Группа материалов 7', 
                 'Группа материалов 8', 'Вариант поставки', 'Запланированная длительность поставки', 'Фактическая длительность поставки', 
                 'День недели', 'Сумма заказа', 'Число позиций', 'Количество обработчиков 7', 'Количество обработчиков 15', 
                 'Количество обработчиков 30', 'Согласование заказа 1', 'Согласование заказа 2', 'Согласование заказа 3', 
                 'Изменение даты поставки 7', 'Изменение даты поставки 15', 'Изменение даты поставки 30', 'Число циклов согласования', 
                 'Количество изменений после согласований', 'Дней между 0_1', 'Дней между 1_2', 'Дней между 2_3', 
                 'Дней между 3_4', 'Дней между 4_5', 'Дней между 5_6', 'Дней между 6_7', 'Дней между 7_8', 
                 'Поставщик-закупщик 1', 'Поставщик-закупщик 2', 'Поставщик-закупщик 3', 'Поставщик-закупщик 4', 
                 'Поставщик-закупщик 5', 'Поставщик-закупщик 6', 'Поставщик-закупщик 7', 'Поставщик-закупщик 8', 
                 'Поставщик-закупщик 9', 'Поставщик-закупщик 10', 'Поставщик-закупщик 11', 'Поставщик-закупщик 12', 
                 'Сумма заказа 1', 'Сумма заказа 2', 'Сумма заказа 3', 'Сумма заказа 4', 'Поставка выполнена раньше 1', 'Перенос даты поставки на бумаге', 
                 'Поставка выполнена раньше 2', 'День недели 1', 'День недели 2', 'Месяц 1-1', 'Месяц 1-2', 'Месяц 2-1', 'Месяц 2-2', 
                 'Месяц 3-1', 'Месяц 3-2']

# Загрузка модели
model = None
prec = None
shap_values = None
explainer = None
explanation = None
shap_values = None
with open(MODEL_SAVE_PATH, 'rb') as file:
    model = pickle.load(file)

with open(PREC_SAVE_PATH, 'rb') as file:
    prec = pickle.load(file)

with open(SHAP_SAVE_PATH, 'rb') as file:
    shap_values = pickle.load(file)

with open(EXPLANATION_SAVE_PATH, 'rb') as file:
    explanation = pickle.load(file)    

with open(EXPLAINER_SAVE_PATH, 'rb') as file:
    explainer = pickle.load(file)    

explanation.feature_names = FEATURE_NAMES

df = pd.read_csv(TEST_DATASET)
df_prec = prec.transform(df)

# @st.cache_resource
def get_explanation(data, n_samples):
    data_p = pd.DataFrame(data.sample(n_samples, random_state=42))
    df_prec = prec.transform(data_p)
    explainer = shap.TreeExplainer(model['model'], df_prec)
    explanation = explainer(df_prec, check_additivity=False)
    return explanation


# @st.cache_resource
def get_shap_values(data, n_samples):
    data_p = pd.DataFrame(data.sample(n_samples, random_state=42))
    df_prec = prec.transform(data_p)
    explainer = shap.TreeExplainer(model['model'], df_prec)
    shap_values = explainer.shap_values(df_prec, check_additivity=False)
    return shap_values, explainer

st.set_page_config(page_title='Home', layout='wide')

# @st.cache_resource
def plot_general_shap():
    with st.spinner('Построение графиков...'):
                n_samples = st.slider('Число записей для анализа', min_value=100, max_value=1000, value=500)
                n_features = st.slider('Число признаков для анализа', min_value=5, max_value=30, value=10)
                st_shap(shap.plots.beeswarm(explanation[:n_samples], max_display=n_features), height=800, width=1280)
                st_shap(shap.plots.decision(explainer.expected_value, shap_values[:n_samples], 
                                            feature_names=FEATURE_NAMES, ignore_warnings=True), height=800, width=1280)

# @st.cache_resource
def plot_deep_shap(n_samples):
    with st.spinner('Построение графиков...'):
        st_shap(shap.force_plot(explainer.expected_value, shap_values[:n_samples], 
                                df_prec.sample(25000, random_state=42)[:n_samples], feature_names=FEATURE_NAMES), 
                                height=600, width=1280)


# @st.cache_resource
def plot_individual():
     with st.spinner('Построение графика...'):
            sample_index = st.number_input(label='Номер записи', min_value=1, max_value=25000)
            st_shap(shap.force_plot(explainer.expected_value, shap_values[sample_index], feature_names=FEATURE_NAMES), height=200, width=1280)
            st_shap(shap.plots.waterfall(explanation[sample_index]), height=800, width=1280)
            st.write(f"Model prediction: {model['model'].predict(df_prec.iloc[sample_index].values.reshape(-1, 103))[0]}")


def main():
    # Header
    st.title('Анализ графиков SHAP')
    st.markdown("""Графики SHAP позволяют интерпретировать прогностическую модель и понять, какие факторы (по мнению модели) 
                ведут к тому или иному исходу. Мы можем проанализировать графики SHAP как для целой выборки, так и для 
                каждой поставки в отдельности.""")
    st.divider()

    # Plots
    with st.spinner('Подготовка отчёта...'):
        # Get shap values
        # explanation = get_explanation(df, df.shape[0])
        # shap_values, explainer = get_shap_values(df, df.shape[0])
        # with open(SHAP_SAVE_PATH, 'wb+') as file:
        #     pickle.dump(shap_values, file)

        # with open(EXPLANATION_SAVE_PATH, 'wb+') as file:
        #     pickle.dump(explanation, file)

        # with open(EXPLAINER_SAVE_PATH, 'wb+') as file:
        #     pickle.dump(explainer, file)

        plot_general_shap()
        n_samples = st.slider('Число записей подробного анализа', min_value=10, max_value=100, value=50)
        plot_deep_shap(n_samples)
        plot_individual()


main()

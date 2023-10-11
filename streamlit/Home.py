# Сервер и система
import streamlit as st
from streamlit.web.cli import main
import streamlit.components.v1 as components
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

# Отображение
import warnings
warnings.filterwarnings('ignore')
sns.set_style('darkgrid')
pd.set_option('display.max_columns', None)


test_df = pd.read_csv(TEST_DATASET)

# Загрузка модели
model = None
prec = None
shap_data = None
with open(MODEL_SAVE_PATH, 'rb') as file:
    model = pickle.load(file)

with open(PREC_SAVE_PATH, 'rb') as file:
    prec = pickle.load(file)

with open(SHAP_SAVE_PATH, 'rb') as file:
    shap_data = pickle.load(file)

explainer = shap_data['explainer']
explanation = shap_data['explanation']
shap_values = shap_data['shap_values']

@st.cache_resource
def get_explanation(data):
    data_p = pd.DataFrame(data.sample(500, random_state=42))
    df_prec = prec.transform(data_p)
    explainer = shap.TreeExplainer(model['model'], df_prec)
    explanation = explainer(df_prec, check_additivity=False)
    return explanation


@st.cache_resource
def get_shap_values(data):
    data_p = pd.DataFrame(data.sample(500, random_state=42))
    df_prec = prec.transform(data_p)
    explainer = shap.TreeExplainer(model['model'], df_prec)
    shap_values = explainer.shap_values(df_prec, check_additivity=False)
    return shap_values, explainer

st.set_page_config('Home')

# Приложение
def main():
    _, col1, col2, _ = st.columns([0.4, 0.3, 0.4, 0.7], gap='small')
    with col1:
        st.image('./streamlit/logo.jpg', width=100)

    with col2:
        st.markdown("<h1 style='text-align: left; color: white;'>UnThinkable</h1>", unsafe_allow_html=True)
    
    st.divider()
    st.text('Платформа для аналитики by Team UnThinkable')
    st.button('Аналитика модели', use_container_width=True)
    st.button('Аналитика данных', use_container_width=True)

    
    df_prec = None
    with st.spinner('Preparing explanation...'):
        with st.spinner('Plotting...'):
                st_shap(shap.plots.waterfall(explanation[3]), height=700, width=700)
                st_shap(shap.plots.beeswarm(explanation), height=700, width=700)
                st_shap(shap.plots.decision(explainer.expected_value, shap_values, feature_names=FEATURE_NAMES), height=700, width=700)
            
        with st.spinner('Plotting...'):
            shap.initjs()
            st_shap(shap.force_plot(explainer.expected_value, shap_values[0], df_prec, feature_names=FEATURE_NAMES), height=200, width=700)
            st_shap(shap.force_plot(explainer.expected_value, shap_values, df_prec, feature_names=FEATURE_NAMES), height=400, width=700)


page = st.sidebar.selectbox('Аналитика', ['Модель', 'Данные'], disabled=True)
main()


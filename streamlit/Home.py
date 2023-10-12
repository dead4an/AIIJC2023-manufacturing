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

# Текст
from helpers.texts import HOME_PAGE_CONTENT


# Пути
ROOT = os.getcwd()
TRAIN_DATASET = os.path.join(ROOT, 'data/train_AIC.csv')
BALANCED_DATASET = os.path.join(ROOT, 'data/balanced_train.csv')
TEST_DATASET = os.path.join(ROOT, 'data/test_AIC.csv')
SUBMISSION_PATH = os.path.join(ROOT, 'submissions/')
MODEL_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_model.dat')
PREC_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_preprocessor.dat')


st.set_page_config('Home')


def main():
    # Header
    _, col1, col2, _ = st.columns([0.4, 0.3, 0.4, 0.7], gap='small')
    with col1:
        st.image('./streamlit/logo.jpg', width=100)

    with col2:
        st.markdown("<h1 style='text-align: left; color: white;'>UnThinkable</h1>", unsafe_allow_html=True)

    st.markdown("<h4 style='text-align: center; color: white;'>Аналитическая платформа by Team UnThinkable</h4>", unsafe_allow_html=True)
    # st.text('Аналитическая платформа by Team UnThinkable')
    st.divider()

    # Content
    st.text(HOME_PAGE_CONTENT)
    
main()
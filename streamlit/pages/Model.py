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


# Пути
ROOT = os.getcwd()
TRAIN_DATASET = os.path.join(ROOT, 'data/train_AIC.csv')
BALANCED_DATASET = os.path.join(ROOT, 'data/balanced_train.csv')
TEST_DATASET = os.path.join(ROOT, 'data/test_AIC.csv')
SUBMISSION_PATH = os.path.join(ROOT, 'submissions/')
MODEL_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_model.dat')
PREC_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_preprocessor.dat')
SHAP_SAVE_PATH = os.path.join(ROOT, 'output/shap.dat')


FEATURE_NAMES = ['Provider_0', 'Provider_1', 'Provider_2', 'Provider_3', 'Provider_4', 'Provider_5', 'Provider_6', 
                 'Provider_7', 'Provider_8', 'Provider_9', 'Provider_10', 'Provider_11', 'Operations_Manager_0', 
                 'Operations_Manager_1', 'Operations_Manager_2', 'Operations_Manager_3', 'Operations_Manager_4', 
                 'Operations_Manager_5', 'Factory', 'Purchasing_Organization_0', 'Purchasing_Organization_1', 
                 'Purchasing_Organization_2', 'Purchasing_Organization_3', 'Purchasing_Organization_4', 
                 'Purchasing_Group_0', 'Purchasing_Group_1', 'Purchasing_Group_2', 'Purchasing_Group_3', 
                 'Purchasing_Group_4', 'Purchasing_Group_5', 'Purchasing_Group_6', 'Purchasing_Group_7', 
                 'Purchasing_Group_8', 'Company_Code_0', 'Company_Code_1', 'Company_Code_2', 'Company_Code_3', 
                 'Company_Code_4', 'EI_0', 'EI_1', 'EI_2', 'EI_3', 'EI_4', 'Material_Group_0', 'Material_Group_1', 
                 'Material_Group_2', 'Material_Group_3', 'Material_Group_4', 'Material_Group_5', 'Material_Group_6', 
                 'Material_Group_7', 'Delivery_Option', 'Duration', 'ETA_Delivery', 'Weekday', 'Sum', 'Position_Count', 
                 'Handlers_7', 'Handlers_15', 'Handlers_30', 'Order_Approval_1', 'Order_Approval_2', 'Order_Approval_3', 
                 'Change_Delivery_Date_7', 'Change_Delivery_Date_15', 'Change_Delivery_Date_30', 'Approval_Cycles', 
                 'Changes_After_Approvals', 'Days_Between_0_1', 'Days_Between_1_2', 'Days_Between_2_3', 
                 'Days_Between_3_4', 'Days_Between_4_5', 'Days_Between_5_6', 'Days_Between_6_7', 'Days_Between_7_8', 
                 'Provider_Purchaser_0', 'Provider_Purchaser_1', 'Provider_Purchaser_2', 'Provider_Purchaser_3', 
                 'Provider_Purchaser_4', 'Provider_Purchaser_5', 'Provider_Purchaser_6', 'Provider_Purchaser_7', 
                 'Provider_Purchaser_8', 'Provider_Purchaser_9', 'Provider_Purchaser_10', 'Provider_Purchaser_11', 
                 'Sum_Fold_0', 'Sum_Fold_1', 'Sum_Fold_2', 'Sum_Fold_3', 'ETC_Difference', 'Change_Difference', 
                 'ETC_Power', 'day_sin', 'day_cos', 'month1_sin', 'month1_cos', 'month2_sin', 'month2_cos', 
                 'month3_sin', 'month3_cos']

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

main()
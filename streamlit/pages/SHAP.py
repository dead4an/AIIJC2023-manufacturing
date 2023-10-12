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

df = pd.read_csv(TEST_DATASET)

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

    df_prec = None
    with st.spinner('Preparing explanation...'):
        # explanation = get_explanation(df)
        # shap_values, explainer = get_shap_values(df)
        # with open(SHAP_SAVE_PATH, 'wb+') as file:
        #     pickle.dump({
        #         'explainer': explainer,
        #         'explanation': explanation,
        #         'shap_values': shap_values
        #     }, file)

        with st.spinner('Plotting...'):
                st_shap(shap.force_plot(explainer.expected_value, shap_values, df_prec, feature_names=FEATURE_NAMES), height=400, width=700)
                st_shap(shap.plots.beeswarm(explanation), height=700, width=700)
                st_shap(shap.plots.decision(explainer.expected_value, shap_values, feature_names=FEATURE_NAMES), height=700, width=700)
            
        with st.spinner('Plotting...'):
            shap.initjs()
            st_shap(shap.force_plot(explainer.expected_value, shap_values[0], df_prec, feature_names=FEATURE_NAMES), height=200, width=700)
            st_shap(shap.plots.waterfall(explanation[3]), height=700, width=700)

# 
# vals = df.iloc[0].values
def diags(vals):
    """ Функция построения диаграмм

    Функция построения диаграмм распределения значений и долей срывов.

    Параметры:
        vals: список, содержащий для каждой фичи по 1 значению, которое подсвечивается на диаграммах этой фичи 
    
    Возвращает: список объектов plotly Figure """

    desc = df.describe()
    figs = []
    mg = 250
    for i in range(len(df.columns[:-1])):
        tgt = vals[i]
        column = df.columns[i]
        uq = len(df[column].unique())
        if uq > mg:
            k = 1 / ((desc[column][7] - desc[column][3]) / mg)
        else:
            k = 100
        df[column] = np.round(df[column] * k) / k
        tgt = round(2 * k, 0) / k
        r = {}
        for i in range(len(df)):
            v, y = df[column][i], df['y'][i]
            if v not in list(r.keys()):
                r[v] = [0, 0]
            r[v][1] += 1
            if y == 1:
                r[v][0] += 1
        r = dict(sorted(r.items()))
        c1 = []
        c2 = []
        x = []
        vb = []
        for i in list(r.keys()):
            x.append(i)
            vb.append(r[i][0] / r[i][1])
            if i == tgt:
                c1.append('#E2D4B7')
                c2.append('#E2D4B7')
            else:
                c1.append('#647AA3')
                c2.append('#E03616')
        sz = ((pd.DataFrame(r).T[1] - np.amin(np.abs(pd.DataFrame(r).T[1]))) / (np.amax(np.abs(pd.DataFrame(r).T[1])) - np.amin(np.abs(pd.DataFrame(r).T[1])))).tolist()
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Распределение значений', 'Доли срывов'))
        fig.add_trace(go.Bar(x=x, y=sz, marker={'color': c1}, showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=x, y=vb, marker={'color': c2}, showlegend=False), row=2, col=1)
        fig.update_layout(height=600, width=1200, title_text=column, template='plotly_dark', plot_bgcolor='#0E1117', )
        fig.update_yaxes(type='log')
        figs.append(fig)
        st.plotly_chart(fig)
        
    return figs
            

main()
# diags(vals, 500)
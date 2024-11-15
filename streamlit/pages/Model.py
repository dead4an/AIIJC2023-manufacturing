# Сервер и система
import streamlit as st
from streamlit.web.cli import main
import os
import pickle
import tracemalloc
import gc

# Визуализация
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Пути
ROOT = os.getcwd()
BALANCED_DATASET = os.path.join(ROOT, 'data/balanced_train.csv')
TEST_DATASET = os.path.join(ROOT, 'data/test_AIC.csv')
TRAIN_DATASET = os.path.join(ROOT, 'data/train_AIC.csv')
SUBMISSION_PATH = os.path.join(ROOT, 'submissions/')
MODEL_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_model.dat')
PREC_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_preprocessor.dat')
SHAP_SAVE_PATH = os.path.join(ROOT, 'output/shap_values.dat')
EXPLANATION_SAVE_PATH = os.path.join(ROOT, 'output/explanation.dat')
EXPLAINER_SAVE_PATH = os.path.join(ROOT, 'output/explainer.dat')

# Названия фич для SHAP
NEW_FEATURE_NAMES = ['Поставщик 1', 'Поставщик 2', 'Поставщик 3', 'Поставщик 4', 'Поставщик 5', 'Поставщик 6', 'Поставщик 7', 
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

STD_FEATURE_NAMES = ['Поставщик', 'Материал', 'Категорийный менеджер', 'Операционный менеджер', 'Завод', 'Закупочная организация', 'Группа закупок', 
                     'Балансовая единица', 'ЕИ', 'Группа материалов', 'Вариант поставки', 'НРП', 'Длительность', 'До поставки', 'Месяц1', 'Месяц2', 
                     'Месяц3', 'День недели 2', 'Сумма', 'Количество позиций', 'Количество', 'Количество обработчиков 7', 'Количество обработчиков 15', 
                     'Количество обработчиков 30', 'Согласование заказа 1', 'Согласование заказа 2', 'Согласование заказа 3', 'Изменение даты поставки 7', 
                     'Изменение даты поставки 15', 'Изменение даты поставки 30', 'Отмена полного деблокирования заказа на закупку', 
                     'Изменение позиции заказа на закупку: изменение даты поставки на бумаге', 'Изменение позиции заказа на закупку: дата поставки', 
                     'Количество циклов согласования', 'Количество изменений после согласований', 'Дней между 0_1', 'Дней между 1_2', 'Дней между 2_3', 
                     'Дней между 3_4', 'Дней между 4_5', 'Дней между 5_6', 'Дней между 6_7', 'Дней между 7_8']

tracemalloc.start()

# Загрузка модели
model = None
prec = None
shap_values = None
explainer = None
explanation = None
shap_values = None
df = None
df_prec = None

st.set_page_config(page_title='Model', layout='wide')

@st.cache_resource
def load_model():
    with open(MODEL_SAVE_PATH, 'rb') as file:
        model = pickle.load(file)
    with open(PREC_SAVE_PATH, 'rb') as file:
        prec = pickle.load(file)

    return model, prec

@st.cache_resource
def prec_data(data):
    return prec.transform(data)

@st.cache_resource
def pdct(df):
    pred = model.predict(df)
    res = pd.DataFrame({'value': pred}, index=pd.DataFrame({'id': df.index.tolist()})['id'])
    return res

@st.cache_resource
def diags(df, vals, mg=200, h=600, w=1200):
    """ Функция построения диаграмм

    Функция построения диаграмм распределения значений и долей срывов.

    Параметры:
        vals: список, содержащий для каждой фичи по 1 значению, которое подсвечивается на диаграммах этой фичи
        mg: Максимальное количество столбцов для отображения
        h: высота plotly plots
        w: ширина plotly plots"""

    nl = []
    sl = df['Количество изменений после согласований'].tolist()
    for i in range(len(df)):
        nl.append(min(1000, sl[i]))
    df['Количество изменений после согласований'] = nl
    desc = df.describe()
    for i in range(len(df.columns[:-1])):
        tgt = vals[i]
        column = df.columns[i]
        uq = len(df[column].unique())
        if uq > mg:
            k = 1 / ((desc[column][7] - desc[column][3]) / mg)
        else:
            k = 100
        df[column] = np.round(df[column] * k) / k
        tgt = round(vals[i] * k, 0) / k
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
        fig.update_layout(height=h, width=w, title_text=column, template='plotly_dark', plot_bgcolor='#E0E0E0')
        st.plotly_chart(fig)

def main():
    st.markdown('# Инференс модели')
    st.markdown("""Этот раздел предназначен для совершения предсказаний.""")
    st.divider()

    tab_one_sample, tab_csv = st.tabs(['Предсказание записи', 'Предсказание CSV'])
    with tab_one_sample:
        col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])
        with col1:
            f1 = st.number_input(label="Поставщик", value=1, min_value=1, max_value=2720, step=1)
            f2 = st.number_input(label="Материал", value=5827, min_value=1, max_value=27439, step=1)
            f3 = st.number_input(label="Категорийный менеджер", value=3, min_value=1, max_value=15, step=1)
            f4 = st.number_input(label="Операционный менеджер", value=1, min_value=1, max_value=38, step=1)
            f5 = st.number_input(label="Завод", value=1, min_value=1, max_value=82, step=1)
            f6 = st.number_input(label="Закупочная организация", value=3, min_value=1, max_value=26, step=1)
            f7 = st.number_input(label="Группа закупок", value=46, min_value=1, max_value=379, step=1)
            f8 = st.number_input(label="Балансовая единица", value=2, min_value=1, max_value=17, step=1)
            f9 = st.number_input(label="ЕИ", value=3, min_value=1, max_value=29, step=1)
            f10 = st.number_input(label="Группа материалов", value=20, min_value=1, max_value=191, step=1)
            f11 = st.number_input(label="Вариант поставки", value=1, min_value=1, max_value=3, step=1)
        with col2:
            f12 = st.number_input(label="НРП", value=1, min_value=0, max_value=100, step=1)
            f13 = st.number_input(label="Длительность", value=74, min_value=0, max_value=600, step=1)
            f14 = st.number_input(label="До поставки", value=74, min_value=0, max_value=1000, step=1)
            f15 = st.number_input(label="Месяц1", value=9, min_value=1, max_value=12, step=1)
            f16 = st.number_input(label="Месяц2", value=11, min_value=1, max_value=12, step=1)
            f17 = st.number_input(label="Месяц3", value=9, min_value=1, max_value=12, step=1)
            f18 = st.number_input(label="День недели 2", value=3, min_value=0, max_value=6, step=1)
            f19 = st.number_input(label="Сумма", value=7.43, step=0.01)
            f20 = st.number_input(label="Количество позиций", value=4, min_value=1, max_value=1000, step=1)
            f21 = st.number_input(label="Количество", value=50.0, min_value=0.0, max_value=1000000.0, step=0.01)
            f22 = st.number_input(label="Количество обработчиков 7", value=8, min_value=1, max_value=100, step=1)
        with col3:
            f23 = st.number_input(label="Количество обработчиков 15", value=8, min_value=1, max_value=100, step=1)
            f24 = st.number_input(label="Количество обработчиков 30", value=8, min_value=1, max_value=100, step=1)
            f25 = st.number_input(label="Согласование заказа 1", value=2, min_value=0, max_value=100, step=1)
            f26 = st.number_input(label="Согласование заказа 2", value=2, min_value=0, max_value=100, step=1)
            f27 = st.number_input(label="Согласование заказа 3", value=2, min_value=0, max_value=100, step=1)
            f28 = st.number_input(label="Изменение даты поставки 7", value=4, min_value=0, max_value=200, step=1)
            f29 = st.number_input(label="Изменение даты поставки 15", value=10, min_value=0, max_value=200, step=1)
            f30 = st.number_input(label="Изменение даты поставки 30", value=12, min_value=0, max_value=200, step=1)
            f31 = st.number_input(label="Отмена полного деблокирования заказа на закупку", value=3, min_value=0, max_value=100, step=1)
            f32 = st.number_input(label="Изменение позиции заказа: на бумаге", value=5, min_value=0, max_value=100, step=1)
            f33 = st.number_input(label="Изменение позиции заказа: дата поставки", value=3, min_value=0, max_value=100, step=1)
        with col4:
            f34 = st.number_input(label="Количество циклов согласования", value=6, min_value=0, max_value=100, step=1) 
            f35 = st.number_input(label="Количество изменений после согласований", value=23, min_value=0, max_value=10000, step=1)
            f36 = st.number_input(label="Дней между 0_1", value=32, min_value=-1, max_value=1000, step=1)
            f37 = st.number_input(label="Дней между 1_2", value=15, min_value=-1, max_value=1000, step=1)
            f38 = st.number_input(label="Дней между 2_3", value=2, min_value=-1, max_value=1000, step=1)
            f39 = st.number_input(label="Дней между 3_4", value=4, min_value=-1, max_value=1000, step=1)
            f40 = st.number_input(label="Дней между 4_5", value=3, min_value=-1, max_value=1000, step=1)
            f41 = st.number_input(label="Дней между 5_6", value=17, min_value=-1, max_value=1000, step=1)
            f42 = st.number_input(label="Дней между 6_7", value=-1, min_value=-1, max_value=1000, step=1)
            f43 = st.number_input(label="Дней между 7_8", value=7, min_value=-1, max_value=1000, step=1)
        
        l = [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, 
             f11, f12, f13, f14, f15, f16, f17, f18, f19, 
             f20, f21, f22, f23, f24, f25, f26, f27, f28, 
             f29, f30, f31, f32, f33, f34, f35, f36, f37, 
             f38, f39, f40, f41, f42, f43]

        sample = pd.DataFrame(l, STD_FEATURE_NAMES)
        
        col1, col2 = st.columns([0.2, 0.5])
        with col1:
            st.write(sample)
        with col2:
            predict = model.predict_proba(sample.T)  * 100
            st.write(f"**Шанс срыва: {predict[0][1]:.2f}%**")
            st.write(f"**Шанс успеха: {predict[0][0]:.2f}%**")

        #container = st.sidebar.container()
        #with container:
        #    predict = model.predict_proba(sample.T) * 100
        #    st.markdown(f"<span style='color: #000000;'>**Шанс срыва: {predict[0][1]:.3f}%**</span>", unsafe_allow_html=True)
        #    st.progress(int(predict[0][1]))

        st.divider()
        if st.button('Аналитика'):
            st.markdown("""Ниже представлены диаграммы распределения значений признаков и долей срывов для значений признаков. 
            По этим графикам можно сравнить выбранное значение признака с остальными значениями 
            этого признака по количеству записей в тренировочном датасете и доле записей со срывами в тренировочном датасете.\n
- Чем больше выделенный столбик на диаграмме распределения значений - тем больше записей в тренировочном датасете содержат выбранное значение данного признака
- Чем больше выделенный столбик на диаграмме долей срывов - тем больше доля записей со срывами среди всех записей тренировочного датасета, содержащих данное значение признака""")
            df = pd.read_csv(TRAIN_DATASET)
            diags(df, l)

    with tab_csv:
        csv_file = st.file_uploader("Загрузите файл CSV", type=["csv"])
        if csv_file is not None:
            df = pd.read_csv(csv_file)
            try:
                res = pdct(df)
                st.write(res)
                ind = st.number_input(label='ID записи', min_value=1, max_value=len(res))
                if ind:
                    st.markdown(f"Предсказание: {res['value'][ind]}")
            except Exception as e:
                st.markdown("<span style='color: #FF4B4B;'>**Ошибка! Попробуйте загрузить другие данные!**</span>", unsafe_allow_html=True)
        

if __name__ == '__main__':
    df = pd.read_csv(TEST_DATASET)
    model, prec = load_model()
        
    main()


usage = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(usage[0] >> 20, usage[1] >> 20)
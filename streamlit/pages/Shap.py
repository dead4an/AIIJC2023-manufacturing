# Сервер и система
import streamlit as st
from streamlit.web.cli import main
import os
import pickle
import tracemalloc
import gc

# Визуализация
import pandas as pd
import shap
from streamlit_shap import st_shap
import pandas as pd

from helpers.texts import BEESWARM_HEADER, BEESWARM_RESULTS, DECISION_HEADER, DECISION_RESULTS

# Пути
ROOT = os.getcwd()
TEST_DATASET = os.path.join(ROOT, 'data', 'test_AIC.csv')
MODEL_SAVE_PATH = os.path.join(ROOT, 'output', 'lgbm_model.dat')
PREC_SAVE_PATH = os.path.join(ROOT, 'output', 'lgbm_preprocessor.dat')

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
test_df = pd.read_csv(TEST_DATASET)


@st.cache_resource
def get_explanation(data, n_samples):
    data_p = pd.DataFrame(data.sample(n_samples, random_state=42))
    df_prec = prec.transform(data_p)
    explainer = shap.TreeExplainer(model['model'], df_prec)
    explanation = explainer(df_prec, check_additivity=False)
    return explanation

@st.cache_resource
def get_shap_values(data, n_samples):
    data_p = pd.DataFrame(data.sample(n_samples, random_state=42))
    df_prec = prec.transform(data_p)
    explainer = shap.TreeExplainer(model['model'], df_prec)
    shap_values = explainer.shap_values(df_prec, check_additivity=False)
    return shap_values, explainer

st.set_page_config(page_title='Shap', layout='wide')

@st.cache_resource
def plot_beeswarm(n_samples, n_features):
    with st.spinner('Построение графиков...'):
                st_shap(shap.plots.beeswarm(explanation[:n_samples] * -1, max_display=n_features), height=800, width=1280)

def plot_decision(n_samples, n_features):
     st_shap(shap.plots.decision(explainer.expected_value * -1, shap_values[:n_samples] * -1, feature_display_range=slice(-1, -n_features, -1),
                                 feature_names=FEATURE_NAMES, ignore_warnings=True, show=False), height=1200, width=1400)

@st.cache_resource
def plot_additivity_force(n_samples):
    with st.spinner('Построение графиков...'):
        st_shap(shap.force_plot(explainer.expected_value * -1, shap_values[:n_samples] * -1, 
                                df_prec.sample(1000, random_state=42)[:n_samples], feature_names=FEATURE_NAMES, 
                                show=False), height=600, width=1400)

@st.cache_resource
def plot_individual(sample_index, n_samples):
     with st.spinner('Построение графика...'):
            st_shap(shap.force_plot(explainer.expected_value * -1, shap_values[sample_index] * -1, feature_names=FEATURE_NAMES, show=False), height=200, width=1280)
            st_shap(shap.plots.waterfall(explanation[sample_index] * -1, show=False, max_display=n_samples), height=800 + 24 * max(0, n_samples - 20), width=1280)
            st.write(f"Model prediction: {model.predict(df.iloc[[sample_index]])}")


def main():
    # Header
    st.markdown('# Анализ графиков SHAP')
    st.markdown("""Графики SHAP помогают интерпретировать прогностическую модель и понять, какие факторы (по мнению модели) 
                ведут к тому или иному исходу. Мы можем проанализировать графики SHAP как для целой выборки, так и для
                каждой поставки в отдельности. Важно понимать, что SHAP достаточно мощный инструмент для анализа модели, однако
                не идеальный, поэтому закономерности выявленные для данных в целом, могут не соблюдаться для некоторых точек данных,
                ввиду влияния множества факторов, которые невозможно полностью учесть.""")
    st.divider()

    tab_beeswarm, tab_decision, tab_additivity, tab_details = st.tabs(['Beeswarm', 'Decision', 'Additivity Force', 'Details'])

    # Beeswarm
    with tab_beeswarm:
        st.markdown(BEESWARM_HEADER)
        st.divider()
        
        n_samples = st.slider('Число записей для анализа', min_value=100, max_value=1000, value=500, key='beeswarm')
        n_features = st.slider('Число признаков для анализа', min_value=5, max_value=30, value=15, key='beeswarm_f')
        
        plot_beeswarm(n_samples, n_features)
        st.divider()

        st.markdown(BEESWARM_RESULTS)

    # Decision 
    with tab_decision:
        st.markdown(DECISION_HEADER)
        st.divider()

        n_samples = st.slider('Число записей для анализа', min_value=100, max_value=1000, value=500, key='decision')
        n_features = st.slider('Число признаков для анализа', min_value=10, max_value=30, value=15, key='decision_f')

        plot_decision(n_samples, n_features)
        st.divider()

        st.markdown(DECISION_RESULTS)
    
    # Additivity Force
    with tab_additivity:
        st.markdown("## **Additive Force**")
        st.markdown("""**Additive Force** - интерактивный график, отображащий влияние совокупности признаков в каждой точке данных.
                    Наведя курсор, вы сможете увидеть, какие какие признаки больше всего повлияли на прогноз модели.\n
- Синий цвет: признак и его значение повышают вероятность негативного прогноза срыва поставки (прогнозируется успех)\n
- Красный цвет: признак и его значение повышают вероятность позитивного прогноза срыва поставки (прогнозируется срыв)""")
        
        n_samples = st.slider('Число записей для анализа', min_value=20, max_value=200, value=50, key='additivity')
        plot_additivity_force(n_samples)

    # Details
    with tab_details:
        st.markdown("## **Waterfall**")
        st.markdown("""**Waterfall** - график, отображающий влияние признаков на конкретное предсказание модели. 
        По графику можно понять, какие признаки сыграли меньшую, а какие - бОльшую роль в предсказании.\n
- Синий цвет: признак и его значение повышают вероятность негативного прогноза срыва поставки (прогнозируется успех)\n
- Красный цвет: признак и его значение повышают вероятность позитивного прогноза срыва поставки (прогнозируется срыв)""")
        sample_index = st.number_input(label='Номер записи', min_value=1, max_value=25000, key='details') - 1
        n_samples = st.slider('Количество признаков', min_value=5, max_value=103, value=10, key='details2')
        plot_individual(sample_index, n_samples)

    # Plots
    # Get shap values
    # explanation = get_explanation(df, df.shape[0])
    # shap_values, explainer = get_shap_values(df, df.shape[0])
    # with open(SHAP_SAVE_PATH, 'wb+') as file:
    #     pickle.dump(shap_values, file)

    # with open(EXPLANATION_SAVE_PATH, 'wb+') as file:
    #     pickle.dump(explanation, file)

    # with open(EXPLAINER_SAVE_PATH, 'wb+') as file:
    #     pickle.dump(explainer, file)
#         st.markdown("""Ниже вы можете указать параметры для графика **Beeswarm**""")
#         st.markdown("""- Число записей для анализа: количество точек данных, которые будут рассмотрены\n
# - Число признаков для анализа: количество признаков, влияние которых будет отображено на графике""")

    # sample_index = st.number_input(label='Номер записи', min_value=1, max_value=25000) - 1
    # plot_individual(sample_index)

    # st.markdown("<div style='width: 100%; height: 150px'></div>", unsafe_allow_html=True)
    gc.collect()

if __name__ == '__main__':
    df = pd.read_csv(TEST_DATASET)
    with open(MODEL_SAVE_PATH, 'rb') as file:
        model = pickle.load(file)

    with open(PREC_SAVE_PATH, 'rb') as file:
        prec = pickle.load(file)
        df_prec = prec.transform(df)

    shap_values, explainer = get_shap_values(test_df, 200)
    explanation = get_explanation(test_df, 200)
    explainer.feature_names = FEATURE_NAMES
        
    main()

usage = tracemalloc.get_traced_memory()
tracemalloc.stop()
print(usage[0] >> 20, usage[1] >> 20)
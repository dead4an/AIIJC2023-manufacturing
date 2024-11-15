# Сервер и система
import streamlit as st
from streamlit.web.cli import main
import os
import tracemalloc

# Визуализация
import pandas as pd
import plotly.express as px
import pandas as pd
import numpy as np


ROOT = os.getcwd()
TRAIN_DATASET = os.path.join(ROOT, 'data/train_AIC.csv')
data = pd.read_csv(TRAIN_DATASET)

st.set_page_config(page_title='Data Analysis', layout='wide')

tracemalloc.start()
@st.cache_resource 
def plot_sum_hist():
    fig = px.histogram(data, x='Сумма', color='y', nbins=200)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

@st.cache_resource
def plot_sum_pie():
    data_temp = data.copy()
    data_temp['Группа'] = data_temp['Сумма'].apply(np.ceil)
    sum_data = data_temp.groupby('Группа').filter(lambda x: len(x) >= 500)
    sum_data['y'] = sum_data['y'].apply(lambda _: 1)
    fig = px.pie(sum_data, values='y', names='Группа', hole=0.4)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

@st.cache_resource
def plot_duration_hist(log=False):
    fig = px.histogram(data, x='Длительность', color='y', log_y=log, nbins=200)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

@st.cache_resource
def plot_eta_delivery_hist(log=False):
    data_t = data.copy()
    data_t['Поставка завершена раньше'] = data_t['Длительность'] - data_t['До поставки']
    fig = px.histogram(data_t[data_t['Поставка завершена раньше'] < 50], x='Поставка завершена раньше', color='y', log_y=log, nbins=250)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

@st.cache_resource
def plot_changes_after_approvals_hist(log=False, anomalies=False):
    if anomalies:
        fig = px.histogram(data[data['Количество изменений после согласований'] > 100], x='Количество изменений после согласований', color='y', log_y=log, nbins=300)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)
        return
    
    fig = px.histogram(data[data['Количество изменений после согласований'] <= 100], x='Количество изменений после согласований', color='y', log_y=log)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

@st.cache_resource
def plot_provider_hist(log=False, others=False):
    if others:
        data_others = data.groupby('Поставщик').filter(lambda x: len(x) < 1000)
        fig = px.histogram(data_others, x='Поставщик', color='y', log_y=log)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)
        return
    
    data_default = data.groupby('Поставщик').filter(lambda x: len(x) >= 1000)
    fig = px.histogram(data_default, x='Поставщик', color='y', log_y=log)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

@st.cache_resource
def plot_provider_pie():
    data_default = data.groupby('Поставщик').filter(lambda x: len(x) >= 1000)
    data_default['y'] = data_default['y'].apply(lambda _: 1)
    fig = px.pie(data_default, values='y', names='Поставщик', hole=0.4)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

@st.cache_resource
def plot_provider_deep():
    st.write('## Надёжные поставщики')
    st.write("""
В этом разделе мы провели анализ надёжных поставщиков, о которых имеется как минимум 500 записей,
а доля сорванных поставок не превышает 5%. """)
    data['Поставка выполнена раньше'] = data['Длительность'] - data['До поставки']
    supplier_counts = data['Поставщик'].value_counts()
    top_suppliers = supplier_counts[supplier_counts > 500].index
    df_top_suppliers = data[data['Поставщик'].isin(top_suppliers)]
    total_orders = df_top_suppliers.groupby('Поставщик')['y'].count()
    failed_orders = df_top_suppliers.groupby('Поставщик')['y'].sum()
    failure_rate = (failed_orders / total_orders) * 100
    failure_rate.sort_values(ascending=False)
    
    zero_failures = failure_rate[failure_rate < 5] 
    ret = zero_failures.index
    df_top_suppliers['Надёжный'] = np.where(df_top_suppliers['Поставщик'].isin(ret), 1, 0)
    df_top_suppliers['y1'] = df_top_suppliers['y'].apply(lambda _: 1)

    st.plotly_chart(px.pie(df_top_suppliers[df_top_suppliers['Надёжный'] == 1], 
                    values='y1', names='Поставщик', hole=0.4, height=600, labels={'y1': 'Число записей'}), 
                    use_container_width=True)
    
    st.markdown("""
На этой круговой диаграмме отображены доли надёжных поставщиков среди всех надёжных поставщиков в исторических данных.\n
Топ 5 надёжных поставщиков по количеству завершённых поставок:
1. Поставщик 14
2. Поставщик 17
3. Поставщик 30
4. Поставщик 27
5. Поставщик 29""")
    
    st.divider()

    st.write('## Количество изменений после согласований')
    st.plotly_chart(px.histogram(data_frame=df_top_suppliers[df_top_suppliers['Количество изменений после согласований'] < 80   ], 
                                 x='Количество изменений после согласований', color='Надёжный', nbins=100), use_container_width=True)
    st.write("""
На этом графике мы можем заметить, что у успешных поставщиков значение признака количество изменений в заказе после согласования в среднем не превышает 20-30 изменений.\n 
**Гипотеза**: Надёжные поставщики имеют более продуманные процессы согласования заказов, а также хорошую коммуникацию с заказчиками. Из этого следует
лучшее понимание требований клиента, эффективное взаимодействие и обмен информацией.\n
**Рекомендация**: Следует изучить подходы и методы успешных поставщиков, которые помогают минимизировать количество изменений после согласований.""")
    st.divider()


    st.write('## Операционный менеджер')
    st.plotly_chart(px.histogram(df_top_suppliers, x='Операционный менеджер', color='Надёжный',
                                 labels={'y1': 'Число записей'}).update_layout(xaxis=dict(tickmode='linear',tick0=1, dtick=1)), 
                                 use_container_width=True)
    
    st.write("""
На этом графике мы видим, что надёжные поставщики чаще всего взаимодействуют с операционными менеджерами c ID:
2, 6. Также они взаимодействуют с операционными менеджерами с ID: 1, 3, 4, 5, 8, 12.\n
**Гипотеза**: Операционные менеджеры 10, 11, 17, 18 низкоэффективны, ввиду недостаточно тщательного планирования процесса поставки,
отсутствия резеврных планов на случай форс-мажора, недостатока коммуникации с поставщиком. Возможно,
что этим операционным менеджерам приходится управлять особо сложными заказами.\n
**Рекомендации**: Необходимо выяснить причины, по которым операционные менеджеры 1, 2, 3, 4, 6, 8, 12
лучше справляются со своими обязанностями. Возможно, другим менеджерам следует перенять их опыт, это
позволит добиться лучшего опыта управления заказами.""")
    st.divider()

    st.write('## Категорийный менеджер')
    st.plotly_chart(px.histogram(df_top_suppliers, x='Категорийный менеджер', color='Надёжный',
                                 labels={'y1': 'Число записей'}).update_layout(xaxis=dict(tickmode='linear',tick0=1, dtick=1)), 
                                 use_container_width=True)
    
    st.write("""
В этом распределении, мы можем заметить, что наибольшая доля успешных поставщиков взаимодействует в основном со следующими
категорийными менеджерами:
- Категорийный менеджер 1
- Категорийный менеджер 2
- Категорийный менеджер 3
- Категорийный менеджер 4
             
**Гипотеза**: Эти категорийные менеджеры лучше занимаются управлением ассортимента, более эффективно налаживают коммуникацию
с поставщиками, а ценова политика, определяемая ими, устраивает обе стороны сделки.\n
**Рекомендации**: Проанализировать работу категорийных менеджеров 1, 2, 3 и 4, выявить отличия от остальных менеджеров. Особое
внимание нужно уделить ценовой политике, определяемой менеджерами.""")
    st.divider()

    st.write('## Закупочная организация')
    st.plotly_chart(px.histogram(df_top_suppliers, x='Закупочная организация', color='Надёжный',
                                 labels={'y1': 'Число записей'}).update_layout(xaxis=dict(tickmode='linear',tick0=1, dtick=1)), 
                                 use_container_width=True)

    col1, col2 = st.columns([0.3, 0.5])
    with col1:
        st.write(df_top_suppliers.groupby('Закупочная организация')[['Сумма', 'Длительность', 'y']].mean().rename({'y': 'Процент срывов'}, axis=1))
    with col2:
        st.write("""
На этом графике видно, что надёжные поставщики чаще всего взаимодействуют с закупочными организациями 
c ID: 1, 2, 11. С закупочными организациями 8, 9, 7, 3, 4, 14, 15 взаимодействуют в основном поставщики, имеющие долю сорванных
поставок выше 5%.\n
**Гипотеза**:  Закупочные организации 1, 2, 11 более эффективны, поскольку выбирают оптимальных поставщиков,
правильно планируют процесс закупок.\n
**Рекомендации**: Подробно исследовать процессы закупок в закупочных организациях с ID: 1, 2, 11.
Изучить возможные причины, по которым закупочные организации 3, 4, 7, 8, 9, 14, 15 в основном взаимодействуют
не с самыми эффективными поставщиками, оптимизировать процессы закупок в этих организациях.""")
    st.divider()

    st.write('## Согласование заказа')
    st.plotly_chart(px.histogram(df_top_suppliers, x='Согласование заказа 1', color='Надёжный',
                                 labels={'y1': 'Число записей'}).update_layout(xaxis=dict(tickmode='linear',tick0=1, dtick=1)), 
                                 use_container_width=True)
    
    st.write("""
Исходя из этого графика распределения, частота нахождения заказа на первой стадии согласования у надёжных поставщиков в среднем варьируется от
0 до 3 раз. Такой же тренд наблюдается для второй и третьей стадии согласования.\n
**Гипотеза**: Заказы, связанные с надёжными поставщиками, могут реже находиться на стадиях согласования по следующим причинам: 
- Доверие к надёжным поставщикам позволяет обеим сторонам быстрее проводить процедуры согласования заказа, поскольку
обе стороны уверены в партнёре.
- Надёжные поставщики зачастую хорошо соблюдают стандарты исполнения своих обязанностей, поэтому компании заказчику реже
приходится проводить дополнительные проверки и вносить правки в заказ.\n
**Рекомендации**: Необходимо собрать подробную информацию о процессах согласования заказов, связанных с надёжными поставщиками, а
также оценить поведение менеджеров при заключении подобных заказов.""")
    st.divider()

    st.write('## Поставка выполнена раньше')
    st.plotly_chart(px.histogram(df_top_suppliers[df_top_suppliers['Поставка выполнена раньше'] < 30], x='Поставка выполнена раньше', color='Надёжный',
                                 labels={'y1': 'Число записей'}), 
                                 use_container_width=True)
    st.write("""
Из графика следует, что надёжные поставщики, как правило, доставляют заказы либо в намеченный срок, либо
от 1 до 10 дней раньше. Поставки выполненные на месяц раньше также имеют высокую долю надёжных поставщиков (54%).
- Надёжные поставщики чаще доставляют свой товар вовремя, либо от 1 до 13 дней заранее.
- Большой процент успешных поставщиков в поставках, завершённых на месяц раньше, может быть связан
с поставками, которые были перенесены на месяц раньше изначального плана.
- Прочие поставщики по могут доставлять свой товар на срок от 14 до 28 дней раньше.
- Важно учитывать, что незапланированно завершённая раньше срока поставка может повлечь за собой дополнительные расходы.""")

@st.cache_resource
def plot_purchaser_hist(log=False, others=False):
    if others:
        data_others = data.groupby('Закупочная организация').filter(lambda x: len(x) < 1000)
        fig = px.histogram(data_others, x='Закупочная организация', color='y', log_y=log)
        st.plotly_chart(fig, theme='streamlit', use_container_width=True)
        return
    
    data_default = data.groupby('Закупочная организация').filter(lambda x: len(x) >= 1000)
    fig = px.histogram(data_default, x='Закупочная организация', color='y', log_y=log)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

@st.cache_resource
def plot_purchaser_pie():
    data_default = data.groupby('Закупочная организация').filter(lambda x: len(x) >= 1000)
    data_default['y'] = data_default['y'].apply(lambda _: 1)
    fig = px.pie(data_default, values='y', names='Закупочная организация', hole=0.4, labels={'y': 'Число записей'})
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

@st.cache_resource
def group_sum(data):
    sum_data = data[['Сумма', 'Количество', 'Длительность', 'y']].copy()
    positive_data = sum_data[sum_data['y'] == 1].mean()
    negative_data = sum_data[sum_data['y'] == 0].mean()

    data_higher = sum_data[sum_data['Сумма'] >= 6]['y']
    data_lower = sum_data[sum_data['Сумма'] < 6]['y']

    sum_data['Группа'] = sum_data['Сумма'].apply(np.ceil)
    sum_data = sum_data.rename(mapper={'Сумма': 'Медиана суммы', 'Длительность': 'Медиана длительности',
                                       'Количество': 'Медиана количества', 'y': 'Процент срыва'}, axis=1)

    col1, col2 = st.columns([0.45, 0.55])
    with col1:
        st.dataframe(sum_data.groupby('Группа').agg({'Медиана суммы': np.median, 'Медиана длительности': np.median,
                                                     'Медиана количества': np.median, 'Процент срыва': lambda x: x.mean() * 100}))
    with col2:
        st.write("- Наиболее низкая доля срывов наблюдается в группах 4-7")
        st.write(f"- Средняя сумма заказа для сорванных поставок: {positive_data['Сумма'].mean().round(decimals=2)}")
        st.write(f"- Средняя сумма заказа для успешных поставок: {negative_data['Сумма'].mean().round(decimals=2)}")
        st.write(f"- Процент срывов для заказов с суммой больше 6: {(data_higher.sum() / data_higher.shape[0]).round(decimals=4) * 100}%")
        st.write(f"- Процент срывов для заказов с суммой меньше 6: {(data_lower.sum() / data_lower.shape[0]).round(decimals=4) * 100}%")
        st.write("""- Процент срывов для заказов на сумму менее 6 значительно ниже, чем для заказов на сумму более 6, 
при этом средняя сумма заказов успешных и сорванных поставок приблизительно одна и та же.""")


def main():
    st.title('Анализ данных')
    st.markdown("""
В этом разделе мы проводим подробный анализ данных. Данные — ресурс, а графики — инструмент, позволяющий получить из них пользу.""")
    st.divider()

    tab_general, tab_deep = st.tabs(['Общий анализ', 'Подробный анализ'])
    # Поставщик
    with tab_deep:
            plot_provider_deep()

    with tab_general:
        selected_feature = st.sidebar.selectbox(label='Признак', 
                                        options=['Поставщик', 'Закупочная организация', 'Сумма заказа', 'Планируемая длительность поставки', 
                                                 'Поставка выполнена раньше', 'Количество изменений после согласований'])
        if selected_feature == 'Поставщик':
            st.markdown('## Поставщик')
            default, default_pie, others = st.tabs(['Основные (Log)', 'Основные (Pie)', 'Прочие (Log)'])
            with default:
                plot_provider_hist(True)
                st.markdown("""На этом графике отображены основные поставщики, о которых имеется как минимум 1.000 записей. 
Мы выбрали 10 самых надёжных поставщиков, а также 5 поставщиков, имеющих высокий процент срыва поставок.""")
                st.divider()
                data_default = data.groupby('Поставщик').filter(lambda x: len(x) >= 1000)
                col1, col2, col3 = st.columns([0.4, 0.2, 0.3])
                with col1:
                    counts = data_default.groupby('Поставщик')['y'].count()
                    counts_positive = data_default.groupby('Поставщик').apply(lambda x: x[x['y'] == 1].count())
                    counts = pd.DataFrame(counts)
                    counts.columns = ['Количество поставок']
                    mean_value = data_default.groupby('Поставщик')['y'].mean()
                    mean_value = ((mean_value) * 100)
                    mean_value = pd.DataFrame(mean_value)
                    mean_value.columns = ['Процент срывов']
                    mean_value['Количество поставок'] = counts['Количество поставок']
                    mean_value['Количество срывов'] = counts_positive.iloc[:, 1]
                    st.write(mean_value.sort_values('Процент срывов'))
                with col2:
                    st.markdown("""
**Топ 10 надёжных поставщиков**
1. Поставщик 30
2. Поставщик 27
3. Поставщик 29
4. Поставщик 14
5. Поставщик 17
6. Поставщик 9
7. Поставщик 26
8. Поставщик 31
9. Поставщик 20
10. Поставщик 21""")
                with col3:
                    st.markdown("""
**Топ 5 ненадёжных поставщиков**
1. Поставщик 7
2. Поставщик 18
3. Поставщик 25
4. Поставщик 11
5. Поставщик 1""")
                
            with default_pie:
                st.markdown("""
    На этой интерактивной круговой диаграмме представлено распределение поставок среди
    основных поставщиков, о которых имеется более 1000 записей.""")
                plot_provider_pie()
                col1, col2, col3 = st.columns([0.4, 0.2, 0.3])
                with col1:
                    counts = data_default.groupby('Поставщик')['y'].count()
                    counts_positive = data_default.groupby('Поставщик').apply(lambda x: x[x['y'] == 1].count())
                    counts = pd.DataFrame(counts)
                    counts.columns = ['Количество поставок']
                    mean_value = data_default.groupby('Поставщик')['y'].mean()
                    mean_value = ((mean_value) * 100)
                    mean_value = pd.DataFrame(mean_value)
                    mean_value.columns = ['Процент срывов']
                    mean_value['Количество поставок'] = counts['Количество поставок']
                    mean_value['Количество срывов'] = counts_positive.iloc[:, 1]
                    st.write(mean_value.sort_values('Процент срывов'))
                with col2:
                    st.markdown("""
**Топ 10 надёжных поставщиков**
1. Поставщик 30
2. Поставщик 27
3. Поставщик 29
4. Поставщик 14
5. Поставщик 17
6. Поставщик 9
7. Поставщик 26
8. Поставщик 31
9. Поставщик 20
10. Поставщик 21""")
                with col3:
                    st.markdown("""
**Топ 5 ненадёжных поставщиков**
1. Поставщик 7
2. Поставщик 18
3. Поставщик 25
4. Поставщик 11
5. Поставщик 1""")
                    
            with others:
                plot_provider_hist(True, True)
                st.markdown("""
В этом разделе представленны прочие поставщики, имеющие менее 1000 записей о поставках.""")
                st.divider()
                provider_id = st.number_input(min_value=1, max_value=2729, label='ID Поставщика')
                data_default = data.groupby('Поставщик').filter(lambda x: len(x) < 1000)
                col1, col2, col3 = st.columns([0.25, 0.3, 0.15])
                with col1:
                    st.write('Статистика поставщиков')
                    counts = data_default.groupby('Поставщик')['y'].count()
                    counts_positive = data_default.groupby('Поставщик').apply(lambda x: x[x['y'] == 1].count())
                    counts = pd.DataFrame(counts)
                    counts.columns = ['Количество поставок']
                    mean_value = data_default.groupby('Поставщик')['y'].mean()
                    mean_value = ((mean_value) * 100)
                    mean_value = pd.DataFrame(mean_value)
                    mean_value.columns = ['Процент срывов']
                    mean_value['Количество поставок'] = counts['Количество поставок']
                    mean_value['Количество срывов'] = counts_positive.iloc[:, 1]
                    st.write(mean_value.sort_values('Поставщик'))

                with col2:
                    st.write(f'Записи о поставщике {provider_id}')
                    st.write(data[data['Поставщик'] == provider_id])

                with col3:
                    st.write(f'Статистика поставщика {provider_id}')
                    counts = data[data['Поставщик'] == provider_id]['y'].count()
                    counts_positive = data[(data['Поставщик'] == provider_id) & (data['y'] == 1)]['y'].count()
                    mean_value = data[data['Поставщик'] == provider_id]['y'].mean()
                    mean_value = ((mean_value) * 100)
                    mean_value = pd.Series({'Поставщик': provider_id, 'Количество поставок': counts,
                                            'Количество срывов': counts_positive, 'Процент срывов': mean_value}, name='Статистика')
                    st.write(mean_value)
            
        # Закупочная организация
        elif selected_feature == 'Закупочная организация':
            with tab_general:
                st.markdown('## Закупочная организация')
                default, default_pie, others = st.tabs(['Основные (Log)', 'Основные (Pie)', 'Прочие (Log)'])
                with default:
                    plot_purchaser_hist(True)
                    st.markdown("""На этом графике отображены основные закупочные организации, о которых имеется как минимум 1.000 записей. 
Мы выбрали 10 самых надёжных закупочных организаций, а также 5 закупочных организаций, имеющих высокий процент срыва поставок.""")

                    data_default = data.groupby('Закупочная организация').filter(lambda x: len(x) >= 1000)
                    col1, col2, col3 = st.columns([0.4, 0.2, 0.3])
                    with col1:
                        counts = data_default.groupby('Закупочная организация')['y'].count()
                        counts_positive = data_default.groupby('Закупочная организация').apply(lambda x: x[x['y'] == 1].count())
                        counts = pd.DataFrame(counts)
                        counts.columns = ['Количество поставок']
                        mean_value = data_default.groupby('Закупочная организация')['y'].mean()
                        mean_value = ((mean_value) * 100)
                        mean_value = pd.DataFrame(mean_value)
                        mean_value.columns = ['Процент срывов']
                        mean_value['Количество поставок'] = counts['Количество поставок']
                        mean_value['Количество срывов'] = counts_positive.iloc[:, 1]
                        st.write(mean_value.sort_values('Процент срывов'))
                    with col2:
                        st.markdown("""
**Топ 5 надёжных закупочных организаций**
1. Закупочная организация 11
2. Закупочная организация 12
3. Закупочная организация 10
4. Закупочная организация 5
5. Закупочная организация 2""")
                    with col3:
                        st.markdown("""
**Топ 5 ненадёжных закупочных организаций**
1. Закупочная организация 15
2. Закупочная организация 13
3. Закупочная организация 9
4. Закупочная организация 17
5. Закупочная организация 4""")
                
                with default_pie:
                    st.markdown("""
На этой интерактивной круговой диаграмме представлено распределение закупок среди
основных закупочных организаций, о которых имеется более 1000 записей.""")
                    plot_purchaser_pie()
                    col1, col2, col3 = st.columns([0.4, 0.2, 0.3])
                    with col1:
                        counts = data_default.groupby('Закупочная организация')['y'].count()
                        counts_positive = data_default.groupby('Закупочная организация').apply(lambda x: x[x['y'] == 1].count())
                        counts = pd.DataFrame(counts)
                        counts.columns = ['Количество закупок']
                        mean_value = data_default.groupby('Закупочная организация')['y'].mean()
                        mean_value = ((mean_value) * 100)
                        mean_value = pd.DataFrame(mean_value)
                        mean_value.columns = ['Процент срывов']
                        mean_value['Количество закупок'] = counts['Количество закупок']
                        mean_value['Количество срывов'] = counts_positive.iloc[:, 1]
                        st.write(mean_value.sort_values('Процент срывов'))
                    with col2:
                        st.markdown("""
**Топ 5 надёжных закупочных организаций**
1. Закупочная организация 11
2. Закупочная организация 12
3. Закупочная организация 10
4. Закупочная организация 5
5. Закупочная организация 2""")
                    with col3:
                        st.markdown("""
**Топ 5 ненадёжных закупочных организаций**
1. Закупочная организация 15
2. Закупочная организация 13
3. Закупочная организация 9
4. Закупочная организация 17
5. Закупочная организация 4""")
                    
                with others:
                    plot_purchaser_hist(True, True)
                    st.markdown("""
В этом разделе представленны прочие закупочные организации, имеющие менее 1000 записей о поставках.""")
                    purchaser_id = st.number_input(min_value=1, max_value=26, label='ID Закупочной организации')
                    data_default = data.groupby('Закупочная организация').filter(lambda x: len(x) < 1000)
                    col1, col2, col3 = st.columns([0.29, 0.26, 0.15])
                    with col1:
                        st.write('Статистика закупочных организаций')
                        counts = data_default.groupby('Закупочная организация')['y'].count()
                        counts_positive = data_default.groupby('Закупочная организация').apply(lambda x: x[x['y'] == 1].count())
                        counts = pd.DataFrame(counts)
                        counts.columns = ['Количество закупок']
                        mean_value = data_default.groupby('Закупочная организация')['y'].mean()
                        mean_value = ((mean_value) * 100)
                        mean_value = pd.DataFrame(mean_value)
                        mean_value.columns = ['Процент срывов']
                        mean_value['Количество закупок'] = counts['Количество закупок']
                        mean_value['Количество срывов'] = counts_positive.iloc[:, 1]
                        st.write(mean_value.sort_values('Закупочная организация'))

                    with col2:
                        st.write(f'Записи о закупочной организации {purchaser_id}')
                        counts = data[data['Закупочная организация'] == purchaser_id].count()
                        counts_positive = data[(data['Закупочная организация'] == purchaser_id) & (data['y'] == 1)].count()
                        counts = pd.DataFrame(counts)
                        counts.columns = ['Количество закупок']
                        mean_value = data[data['Закупочная организация'] == purchaser_id]
                        mean_value = ((mean_value) * 100)
                        mean_value = pd.DataFrame(mean_value)
                        mean_value['Количество закупок'] = counts['Количество закупок']
                        mean_value['Количество срывов'] = counts_positive
                        st.write(mean_value)

                    with col3:
                        st.write(f'Статистика закупочной организации {purchaser_id}')
                        counts = data[data['Закупочная организация'] == purchaser_id]['y'].count()
                        counts_positive = data[(data['Закупочная организация'] == purchaser_id) & (data['y'] == 1)]['y'].count()
                        mean_value = data[data['Закупочная организация'] == purchaser_id]['y'].mean()
                        mean_value = ((mean_value) * 100)
                        mean_value = pd.Series({'Закупочная организация': purchaser_id, 'Количество закупок': counts,
                                                'Количество срывов': counts_positive, 'Процент срывов': mean_value}, name='Статистика')
                        st.write(mean_value)

        # Сумма
        elif selected_feature == 'Сумма заказа':
            st.markdown('## Сумма заказа')
            tab_hist, tab_pie = st.tabs(['Столбчатая диаграмма', 'Круговая диаграмма'])
            with tab_hist:
                plot_sum_hist()

                st.markdown("""
По графику распределения суммы заказов, можно сказать, что с ростом суммы заказа растёт и шанс срыва поставки.
Такое может быть связано с повышенными требованиями к заказу. Для оптимизации дорогих заказов, стоит выбирать более
надёжных поставщиков, а также тщательнее подходить к процессу согласования заказа.
""")    

            with tab_pie:
                plot_sum_pie()
                st.markdown("""
Из круговой диаграммы групп видно, что доминируют следующие группы заказов:
1. Группа 6 (Сумма заказа в диапазоне 5-6)
2. Группа 7 (Сумма заказа в диапазоне 7-8)
3. Группа 8 (Сумма заказа в диапазоне 8-9)           
4. Группа 5 (Сумма заказа в диапазоне 8-9)
""")    

            st.divider()
            group_sum(data)

        # Длительность
        elif selected_feature == 'Планируемая длительность поставки':
            st.markdown('## Планируеммая длительность поставки')
            default, log = st.tabs(['Стандартный', 'Log'])
            with default:
                plot_duration_hist()
            with log:
                plot_duration_hist(True)

            st.markdown("""
На данном графике распределения мы видим, что большая доля сорванных заказов приходится на краткосрочные поставки.
Однако долгосрочные поставки длительностью более 250 дней имеют меньшую долю срывов, а поставки длительностью 
более 350 дней не имеют срывов вовсе. Такое может быть связано с:
- Более ответственным планированием долгосрочных поставок.
- Долгосрочные поставки связаны с более надёжными поставщиками.
- Срыв долгосрочных поставок невыгоден обеим сторонам, даже в случае несоответствия требованиям заказчика.""")

        # До поставки
        elif selected_feature == 'Поставка выполнена раньше':
            st.markdown('## Поставка выполнена раньше')
            default, log = st.tabs(['Стандартный', 'Log'])
            with default:
                plot_eta_delivery_hist()
            with log:
                plot_eta_delivery_hist(True)

            st.markdown("""
На данном графике распределения мы видим, что поставки, завершённые точно в срок имеют большее количество срывов,
чем поставки, завершённые хотя бы на один день раньше. Это можно связать со следующими факторами:
- Раньше намеченного срока поставки совершают в основном надёжные поставщики, доля срывов которых не превышает 5-10%.
- Удовлетворённость заказачика, ведь поставка, завершённая раньше срока, позволяет более уверенно распоряжаться ресурсами.
- Возможно, что в производственном процессе существуют этапы, требующие редких и дорогостоящих материалов, либо чрезвычайно необходимых для производства, 
поэтому такие поставки могут быть критически важными.""")

        # Количество изменений после согласований
        elif selected_feature == 'Количество изменений после согласований':
            st.markdown('## Количество изменений после согласований')
            default, log, anomaly = st.tabs(['Стандартный', 'Log', 'Аномалии'])
            with default:
                plot_changes_after_approvals_hist()
            with log:
                plot_changes_after_approvals_hist(True)
            with anomaly:
                plot_changes_after_approvals_hist(False, True)
            
            st.markdown("""
Здесь мы можем наблюдать, что доля успешно завершённых поставок падает с ростом количества изменений после согласования заказа.
Большое количество изменений после согласованй может говорить о:
- Недостаточной предусмотрительности двух сторон на стадии согласования.
- Ненадёжности процессов поставок у компании поставщика.
Также можно предположить, что наличие хотя бы одного изменения в заказе влечёт за собой цепочку последующих изменений.""")

    st.markdown("<div style='width: 100%; height: 150px'></div>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
    usage = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(usage[0] >> 20, usage[1] >> 20)

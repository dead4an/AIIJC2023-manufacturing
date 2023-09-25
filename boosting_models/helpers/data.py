# Модули
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from category_encoders import BinaryEncoder


# Категориальные признаки подлежащие кодированию BinaryEncoder
CAT_FEATURES = ['Purchasing_Organization', 'Company_Code', 'Provider', 'Provider_Purchaser',
                'Operations_Manager', 'Sum_Fold', 'Material_Group', 'Purchasing_Group', 'EI']

# Числовые признаки подлежащие масштабированию RobustScaler
SCALE_FEATURES = ['Position_Count', 'Duration', 'ETC_Delivery', 'Changes_After_Approvals',
                  'Order_Approval_1', 'Order_Approval_2', 'Order_Approval_3', 'Sum',
                  'Change_Delivery_Date_7', 'Change_Delivery_Date_15', 'Change_Delivery_Date_30',
                  'Approval_Cycles', 'Handlers_7', 'Handlers_15', 'Handlers_30', 'Days_Between_0_1',
                  'Days_Between_1_2', 'Days_Between_2_3', 'Days_Between_3_4', 'Days_Between_4_5',
                  'Days_Between_5_6', 'Days_Between_6_7', 'Days_Between_7_8', 'ETC_Difference', 'ETC_Power']

# Признаки, не используемые в ходе обучения и предсказания
DROP_FEATURES = ['Material', 'Cancel_Complete_Release', 'Month1', 'Month2', 'Month3', 
                 'Delivery_Date', 'Change_on_Paper', 'Amount', 'Category_Manager', 'NRP']

# Новые именования признаков (для совместимости со всеми моделями)
RENAME_COLS = ['Provider', 'Material', 'Category_Manager', 'Operations_Manager',
               'Factory', 'Purchasing_Organization', 'Purchasing_Group',
               'Company_Code', 'EI', 'Material_Group', 'Delivery_Option',
               'NRP', 'Duration', 'ETC_Delivery', 'Month1', 'Month2', 'Month3',
               'Weekday', 'Sum', 'Position_Count', 'Amount', 'Handlers_7',
               'Handlers_15', 'Handlers_30', 'Order_Approval_1', 'Order_Approval_2',
               'Order_Approval_3', 'Change_Delivery_Date_7', 'Change_Delivery_Date_15',
               'Change_Delivery_Date_30', 'Cancel_Complete_Release', 'Change_on_Paper',
               'Delivery_Date', 'Approval_Cycles', 'Changes_After_Approvals',
               'Days_Between_0_1', 'Days_Between_1_2', 'Days_Between_2_3', 'Days_Between_3_4',
               'Days_Between_4_5', 'Days_Between_5_6', 'Days_Between_6_7', 'Days_Between_7_8']


# Предобработчик
class DataPreprocessor(BaseEstimator, TransformerMixin):
    """ Предобработчик данных. 

    Класс предобработчика данных, совмещающий бинарную кодировку 
    категориальных признаков и масштабирование числовых признаков.
    Наследуется от BaseEstimator и TransformerMixin из модуля
    base библиотеки scikit-learn для совместимости с пайплайнами
    (Pipeline из модуля sklearn.pipeline)

    Параметры:
        cat_features: Список категориальных признаков, которые 
            будут обработаны с помощью бинарного кодировщика.
        scale_features: Список числовых признаков, которые 
            будут обработаны с помощью Robust Scaler.
        drop_features: Список признаков, которые будут откинуты и
            не участвуют в процессах обучения и предсказания.
        rename_cols: Новые именования признаков. Требуется, если
            в признаках содержатся символы, не поддерживаемые
            моделью. 

    Функции:
        fit: Обучает предобработчики для дальнейшего использования.
        transform: Трансформирует датасет для дальнейшего использования. """

    def __init__(self, encode_categorical=True) -> None:
        # Инициализация атрибутов
        self.encode_categorical = encode_categorical

        # Инициализация предобработчиков
        self.bin_encoder = BinaryEncoder(cols=CAT_FEATURES)
        self.robust_scaler = RobustScaler()

    def fit(self, X: pd.DataFrame, y: pd.DataFrame | None) -> object:
        """ Обучение предобработчика.

        Обучает предобработчики, на основе датасета и дополнительных признаков,
        которые выводятся из уже имеющихся (экстракция признаков).

        Параметры: 
            X: Экземпляр pandas.DataFrame, содержащий независимые переменные.
            y: Экземпляр pandas.DataFrame, содержащий зависимые переменные. 

        Пример:
            data_preprocessor = DataPreprocessor(cat_features, scale_features, 
                                                 drop_features, rename_cols)
            data_preprocessor.fit(X_train, y_train)

        Возвращает обученный предобработчик."""

        # Создаём копию датасета, чтобы не изменять исходный
        X_ = X.copy()
        X_.columns = RENAME_COLS

        X_['Weekday'] += 1

        # Экстракция признаков
        X_['Provider_Purchaser'] = [f'{x}_{y}' for x, y in zip(X_['Provider'].values,
                                                               X_['Purchasing_Organization'].values)]
        X_['Sum_Fold'] = X_['Sum'].apply(lambda x: int(x) % 10)
        X_['ETC_Difference'] = X_['Duration'] - X_['ETC_Delivery']
        X_['Change_Difference'] = X_['Delivery_Date'] - X_['Change_on_Paper']
        X_['ETC_Power'] = X_['ETC_Difference'] ^ 2
        
        # Добавляем тригонометрические значения временных признаков
        X_['day_sin'] = np.sin(np.pi * 2 * X_['Weekday'] / 7)
        X_['day_cos'] = np.cos(np.pi * 2 * X_['Weekday'] / 7)
        X_['month1_sin'] = np.sin(np.pi * 2 * X_['Month1'] / 12)
        X_['month1_cos'] = np.cos(np.pi * 2 * X_['Month1'] / 12)
        X_['month2_sin'] = np.sin(np.pi * 2 * X_['Month2'] / 12)
        X_['month2_cos'] = np.cos(np.pi * 2 * X_['Month2'] / 12)
        X_['month3_sin'] = np.sin(np.pi * 2 * X_['Month3'] / 12)
        X_['month3_cos'] = np.cos(np.pi * 2 * X_['Month3'] / 12)

        # Нормализация числовых признаков
        self.robust_scaler.fit(X_[SCALE_FEATURES])

        # Кодировка категориальных признаков
        if self.encode_categorical:
            X_ = self.bin_encoder.fit_transform(X_)

        # Дроп неиспользуемых признаков
        X_ = X_.drop(DROP_FEATURES, axis=1)

        return self

    def transform(self, X) -> pd.DataFrame:
        """ Трансформирование датасета.

        Трансформирует датасет для дальнешего использования моделью.
        Требует предварительного обучения с помощью метода fit.

        Параметры: 
            X: Экземпляр pandas.DataFrame, содержащий независимые переменные. 

        Возвращает трансформированный датасет (экземпляр pandas.DataFrame), 
        готовый для использования моделью.

        Пример:
            data_preprocessor = DataPreprocessor(cat_features, scale_features, 
                                                 drop_features, rename_cols)
            data_preprocessor.fit(X_train, y_train)
            X_preprocessed = data_preprocessor.transform(X_test)"""

        # Создаём копию датасета, чтобы не изменять исходный
        X_ = X.copy()
        X_.columns = RENAME_COLS

        X_['Weekday'] += 1

        # Экстракция фич
        X_['Provider_Purchaser'] = [f'{x}_{y}' for x, y in zip(X_['Provider'].values, 
                                                               X_['Purchasing_Organization'].values)]
        X_['Sum_Fold'] = X_['Sum'].apply(lambda x: int(x) % 10)
        X_['ETC_Difference'] = X_['Duration'] - X_['ETC_Delivery']
        X_['Change_Difference'] = X_['Delivery_Date'] - X_['Change_on_Paper']
        X_['ETC_Power'] = X_['ETC_Difference'] ^ 2

        # Временные фичи
        X_['day_sin'] = np.sin(np.pi * 2 * X_['Weekday'] / 7)
        X_['day_cos'] = np.cos(np.pi * 2 * X_['Weekday'] / 7)
        X_['month1_sin'] = np.sin(np.pi * 2 * X_['Month1'] / 12)
        X_['month1_cos'] = np.cos(np.pi * 2 * X_['Month1'] / 12)
        X_['month2_sin'] = np.sin(np.pi * 2 * X_['Month2'] / 12)
        X_['month2_cos'] = np.cos(np.pi * 2 * X_['Month2'] / 12)
        X_['month3_sin'] = np.sin(np.pi * 2 * X_['Month3'] / 12)
        X_['month3_cos'] = np.cos(np.pi * 2 * X_['Month3'] / 12)

        # Нормализация
        X_[SCALE_FEATURES] = self.robust_scaler.transform(X_[SCALE_FEATURES])

        # Категориальные фичи
        if self.encode_categorical:
            X_ = self.bin_encoder.transform(X_)

        X_ = X_.drop(DROP_FEATURES, axis=1)

        return X_

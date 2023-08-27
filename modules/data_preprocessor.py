import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
from category_encoders import BinaryEncoder


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """ Предобработчик данных """
    def __init__(self, cat_features, scale_features,
                 drop_features, rename_cols, transform_train=True):
        self.transform_train = transform_train
        self.cat_features = cat_features

        self.bin_encoder = BinaryEncoder(cols=cat_features)
        self.robust_scaler = RobustScaler()

        self.rename_cols = rename_cols

        self.drop_features = drop_features
        self.scale_features = scale_features

    def fit(self, X, y=None):
        # Создаём копию датасета
        X_ = X.copy()
        X_.columns = self.rename_cols

        # Временные фичи
        X_['Weekday'] += 1
        X_['day_sin'] = np.sin(np.pi * 2 * X_['Weekday'] / 7)
        X_['day_cos'] = np.cos(np.pi * 2 * X_['Weekday'] / 7)
        X_['month1_sin'] = np.sin(np.pi * 2 * X_['Month1'] / 12)
        X_['month1_cos'] = np.cos(np.pi * 2 * X_['Month1'] / 12)
        X_['month2_sin'] = np.sin(np.pi * 2 * X_['Month2'] / 12)
        X_['month2_cos'] = np.cos(np.pi * 2 * X_['Month2'] / 12)
        X_['month3_sin'] = np.sin(np.pi * 2 * X_['Month3'] / 12)
        X_['month3_cos'] = np.cos(np.pi * 2 * X_['Month3'] / 12)

        # Экстракция фич
        X_['Provider Purchaser'] = [f'{x}_{y}' for x, y in zip(X_['Provider'].values, X_['Purchasing Organization'].values)]
        X_['Provider Delivery option'] = [f'{x}_{y}' for x, y in zip(X_['Provider'].values, X_['Delivery Option'].values)]
        X_['Sum Fold'] = X_['Sum'].apply(lambda x: int(x % 10))
        
        # Нормализация
        self.robust_scaler.fit(X_[self.scale_features])

        # Категориальные фичи        
        X_ = self.bin_encoder.fit_transform(X_)

        X_ = X_.drop(self.drop_features, axis=1)
        
        return self
    
    def transform(self, X):
        # Создаём копию датасета
        X_ = X.copy()
        X_.columns = self.rename_cols

        # Временные фичи
        X_['Weekday'] += 1
        X_['day_sin'] = np.sin(np.pi * 2 * X_['Weekday'] / 7)
        X_['day_cos'] = np.cos(np.pi * 2 * X_['Weekday'] / 7)
        X_['month1_sin'] = np.sin(np.pi * 2 * X_['Month1'] / 12)
        X_['month1_cos'] = np.cos(np.pi * 2 * X_['Month1'] / 12)
        X_['month2_sin'] = np.sin(np.pi * 2 * X_['Month2'] / 12)
        X_['month2_cos'] = np.cos(np.pi * 2 * X_['Month2'] / 12)
        X_['month3_sin'] = np.sin(np.pi * 2 * X_['Month3'] / 12)
        X_['month3_cos'] = np.cos(np.pi * 2 * X_['Month3'] / 12)

        # Экстракция фич
        X_['Provider Purchaser'] = [f'{x}_{y}' for x, y in zip(X_['Provider'].values, X_['Purchasing Organization'].values)]
        X_['Provider Delivery option'] = [f'{x}_{y}' for x, y in zip(X_['Provider'].values, X_['Delivery Option'].values)]
        X_['Sum Fold'] = X_['Sum'].apply(lambda x: x % 10)

        # Нормализация
        X_[self.scale_features] = self.robust_scaler.transform(X_[self.scale_features])

        # Категориальные фичи
        X_ = self.bin_encoder.transform(X_)

        X_ = X_.drop(self.drop_features, axis=1)

        return X_

# Модули
import os
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Пути
ROOT = os.getcwd()
MODEL_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_model.dat')
TEST_DATASET = os.path.join(ROOT, 'data/test_AIC.csv')
SUBMISSION_PATH = os.path.join(ROOT, 'output/')
test_df = pd.read_csv(TEST_DATASET)


def save_submission(preds: list | pd.DataFrame | pd.arrays.PandasArray, subname: str) -> None:
    """ Функция сохранения предсказаний. """
    subname = os.path.join(SUBMISSION_PATH, f'{subname}.csv')
    submit_df = pd.DataFrame({'id': test_df.index, 'value': preds})
    submit_df.to_csv(subname, index=False)


# Загрузка модели (дамп модели из boosting_models/lightgbm.ipynb)
model = None
with open(MODEL_SAVE_PATH, 'rb') as file:
    model = pickle.load(file)

# Сохранение сабмита в output/submission.csv
preds = model.predict(test_df)
save_submission(preds, 'submission')
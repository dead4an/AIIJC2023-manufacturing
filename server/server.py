import os
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, render_template


# Пути
ROOT = os.getcwd()
TRAIN_DATASET = os.path.join(ROOT, 'data/train_AIC.csv')
TEST_DATASET = os.path.join(ROOT, 'data/test_AIC.csv')
MODEL_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_model.dat')

# Загрузка модели
model = None
with open(MODEL_SAVE_PATH, 'rb') as file:
    model = pickle.load(file)

# Загрузка данных
test_df = pd.read_csv(TRAIN_DATASET)

app = Flask(__name__, template_folder='template', static_folder='static')
@app.route('/')
def home_page():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

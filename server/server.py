import os
import pickle
import pandas as pd
from helpers.data import RENAME_COLS
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, render_template


# Пути
ROOT = os.getcwd()
MODEL_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_model.dat')

# Загрузка модели
model = None
with open(MODEL_SAVE_PATH, 'rb') as file:
    model = pickle.load(file)

# Приложение Flask
app = Flask(__name__, template_folder='template', static_folder='static')
@app.route('/', methods=['POST', 'GET'])
def home_page():
    if request.method == 'POST':
        data = request.form.to_dict()
        df = pd.DataFrame(data=[[float(value[1]) for value in data.items()]], columns=RENAME_COLS)
        pred = model.predict(df)
        print(f'Predicted class: {pred[0]}')
        return render_template('index.html')
    
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)

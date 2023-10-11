import os
import pickle
import pandas as pd
from helpers.data import RENAME_COLS
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, request, render_template

import shap
from plotly.offline import init_notebook_mode
import matplotlib.pyplot as plt


# Пути
ROOT = os.getcwd()
MODEL_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_model.dat')
PREC_SAVE_PATH = os.path.join(ROOT, 'output/lgbm_preprocessor.dat')
TEST_DATASET = os.path.join(ROOT, 'data/test_AIC.csv')
SUBMISSION_PATH = os.path.join(ROOT, 'submissions/submission_best.csv')

test_df = pd.read_csv(TEST_DATASET)

# Загрузка модели
model = None
prec = None
with open(MODEL_SAVE_PATH, 'rb') as file:
    model = pickle.load(file)

with open(PREC_SAVE_PATH, 'rb') as file:
    prec = pickle.load(file)

# Приложение Flask
app = Flask(__name__, template_folder='template', static_folder='static')
@app.route('/', methods=['POST', 'GET'])
def home_page():
    if request.method == 'POST':        
        data = request.form.to_dict()
        df = pd.DataFrame(data=[[float(value[1]) for value in data.items()]], columns=RENAME_COLS)
        pred = pd.Series(model.predict(df))
        df_prec = prec.transform(df)
        explainer = shap.TreeExplainer(model['model'], pd.concat([df_prec, pred], axis=1))
        shap_values = explainer(df_prec, pred)
        shap.waterfall_plot(shap_values[0], show=False);
        plt.savefig(os.path.join(ROOT, 'server/static/shap.png'))
        print(f'Predicted class: {pred[0]}')
        return render_template('index.html')
    
    else:
        return render_template('index.html')
    

@app.route('/shap', methods=['POST', 'GET'])
def shap_page():
    explainer = shap.TreeExplainer(model['model'], test_df)
    shap_values = explainer.shap_values(test_df)
    f_plot = shap.force_plot(explainer.expected_value, shap_values, test_df)
    return render_template(f_plot.html())


if __name__ == '__main__':
    app.run(debug=False)

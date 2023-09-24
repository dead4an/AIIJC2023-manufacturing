import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna as opt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import classification_report 
from .data import DataPreprocessor


def valid_predictions(y_true, y_predicted) -> None:
    # Получаем число TN, FP, FN, TP (матрица ошибок)
    confusion_matrix = confusion_matrix(y_true, y_predicted)
    confusion_matrix = pd.DataFrame(data=confusion_matrix, columns=['0', '1'],  
                             index=['0', '1'])
    
    # Отображение матрицы ошибок
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()

    # Отчёт по классификации
    print(classification_report(y_true, y_predicted, digits=5))

    # Получаем F1 и ROC AUC метрики
    print(f1_score(y_true, y_predicted, average='macro'))
    print(roc_auc_score(y_true, y_predicted))


def objective_f1_macro(model: object, parameter_names: dict, X: pd.DataFrame, 
                       y: pd.DataFrame, trial: opt.Trial) -> float:
    """ Функция оптимизации F1 (macro).
    
    Использует библиотеку optuna для подбора параметров из
    заданного диапазона. В качестве оптимизируемой метрики 
    выступает F1 (macro).
    
    Параметры:
        trial: экзмепляр optuna.Trial, представляющей собой
        историю оптимизации целевой функции.
        
    Пример:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100)
        
    Возвращает F1 (macro) метрику для подобранных параметров."""
    
    # Параметры
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1)
    n_estimators = trial.suggest_int('n_estimators', 500, 3000)
    max_depth = trial.suggest_int('max_depth', 6, 32)
    max_bin = trial.suggest_int('max_bin', 32, 200),
    num_leaves = trial.suggest_int('num_leaves', 32, 300)
    reg_lambda = trial.suggest_float('reg_lambda', 0.01, 1)

    parameters = {}
    for parameter in parameter_names:
        if parameter == 'learning_rate':
            parameters.update({parameter: learning_rate})
        elif parameter == 'n_estimators':
            parameters.update({parameter: n_estimators})
        elif parameter == 'max_depth':
            parameters.update({parameter: max_depth})
        elif parameter == 'max_bin':
            parameters

    # Модель
    data_preprocessor = DataPreprocessor()
    model.set_parameters(**)

    pipeline = Pipeline([
        ('data_preproc', data_preprocessor),
        ('model', model)
    ])
    
    cv_score = cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=5), scoring='f1_macro', n_jobs=-1)
    accuracy = cv_score.mean()

    return accuracy

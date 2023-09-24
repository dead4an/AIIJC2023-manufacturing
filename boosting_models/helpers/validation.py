import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import optuna as opt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, make_scorer
from sklearn.pipeline import Pipeline
from .data import DataPreprocessor


def valid_predictions(y_true, y_predicted) -> None:
    # Получаем число TN, FP, FN, TP (матрица ошибок)
    cm = confusion_matrix(y_true, y_predicted)
    cm = pd.DataFrame(data=cm, columns=['0', '1'],  
                             index=['0', '1'])
    
    # Отображение матрицы ошибок
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.show()

    # Отчёт по классификации
    print(classification_report(y_true, y_predicted, digits=5))

    # Получаем F1 и ROC AUC метрики
    print(f1_score(y_true, y_predicted, average='macro'))
    print(roc_auc_score(y_true, y_predicted))


def objective_f1_macro(trial: opt.Trial) -> float:
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

    # Модель
    data_preprocessor = DataPreprocessor(encode_categorical)
    model = LGBMClassifier()

    pipeline = Pipeline([
        ('data_preproc', data_preprocessor),
        ('model', model)
    ])
    
    cv_score = cross_val_score(pipeline, X, y, cv=StratifiedKFold(n_splits=5), scoring='f1_macro', n_jobs=-1)
    accuracy = cv_score.mean()

    return accuracy


def objective_confusion_matrix(model: object, parameter_names: dict, X: pd.DataFrame, 
                       y: pd.DataFrame, trial: opt.Trial, encode_categorical=True) -> int:
    """ Функция оптимизации FP и FN значений матрицы ошибок.
    
    Использует библиотеку optuna для подбора параметров из
    заданного диапазона. В качестве оптимизируемой метрики 
    выступает сумма FP и FN значений матрицы ошибок.
    
    Параметры:
        trial: Экзмепляр optuna.Trial, представляющей собой
            историю оптимизации целевой функции.
        
    Пример:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_confusion_matrix, n_trials=100)
        
    Возвращает сумму FP и FN значений матрицы ошибок для подобранных параметров."""
    
    # Параметры
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1)
    n_estimators = trial.suggest_int('n_estimators', 500, 3000)
    max_depth = trial.suggest_int('max_depth', 6, 32)
    max_bin = trial.suggest_int('max_bin', 32, 200),
    num_leaves = trial.suggest_int('num_leaves', 32, 300)
    reg_lambda = trial.suggest_float('reg_lambda', 0.01, 1)


    # Модель
    data_preprocessor = DataPreprocessor(encode_categorical)

    pipeline = Pipeline([
        ('data_preproc', data_preprocessor),
        ('model', model)
    ])
    
    def confusion_matrix_score(y_true, y_pred):
        _, fp, fn, _ = confusion_matrix([0, 1, 0, 1], [1, 1, 1, 0]).ravel()
        return fp + fn
    
    scorer = make_scorer(confusion_matrix_score)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring=scorer)

    return cv_scores.mean()

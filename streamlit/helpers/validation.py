# Модулей
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, accuracy_score, ConfusionMatrixDisplay


def valid_predictions(y_true, y_pred, classes) -> None:
    """ Выводит метрики модели.

    Выводит следующие метрики:
        Матрица ошибок;
        Отчёт классификации с метриками модели;
        F1-macro;
        ROC-AUC score;
        Accuracy.
    
    Параметры:
        y_true: Экземпляр pandas.DataFrame, содержащий истинные метки классов.
        y_pred: Экземпляр pandas.DataFrame, содержащий предсказанные метки классов.
        classes: Список, содержащий названия предсказываемых классов."""
    
    # Строим матрицу ошибок
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Отображение матрицы ошибок
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot()
    plt.show()

    # Отчёт по классификации
    print(classification_report(y_true, y_pred, digits=5))

    # Получаем F1, ROC AUC и accuracy метрики
    print(f"F1-macro: {f1_score(y_true, y_pred, average='macro')}")
    print(f'ROC-AUC score: {roc_auc_score(y_true, y_pred)}')
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')

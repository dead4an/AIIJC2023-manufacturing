import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, accuracy_score, ConfusionMatrixDisplay


def valid_predictions(y_true, y_pred, classes) -> None:
    # Строим матрицу ошибок
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Отображение матрицы ошибок
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    disp.plot()
    plt.show()

    # Отчёт по классификации
    print(classification_report(y_true, y_pred, digits=5))

    # Получаем F1 и ROC AUC метрики
    print(f"F1-macro: {f1_score(y_true, y_pred, average='macro')}")
    print(f'ROC-AUC score: {roc_auc_score(y_true, y_pred)}')
    print(f'Accuracy: {accuracy_score(y_true, y_pred)}')

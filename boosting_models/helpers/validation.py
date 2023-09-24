import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, make_scorer


def valid_predictions(y_true, y_predicted) -> None:
    # Строим матрицу ошибок
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

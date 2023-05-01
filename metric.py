
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class Metric:
    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        metric = {}
        metric.update({'Accuracy': round(accuracy_score(y_true, y_pred), 4)})
        metric.update({'Precision': round(precision_score(y_true, y_pred), 4)})
        metric.update({'ReCall': round(recall_score(y_true, y_pred), 4)})
        metric.update({'F1': round(f1_score(y_true, y_pred), 4)})
        metric.update({'AUC': round(roc_auc_score(y_true, y_pred), 4)})


        return metric
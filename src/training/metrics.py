from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report


def get_metrics(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    return f1, precision, recall, acc, report
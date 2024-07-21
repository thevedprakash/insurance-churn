# This is logistic regression model

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report

def train(X_train, y_train):
    model = LogisticRegression(max_iter=20000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report
    }

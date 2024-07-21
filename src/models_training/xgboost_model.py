# This is for Xgboost model

import xgboost as xgb
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix, classification_report

def train(X_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), random_state=42)
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

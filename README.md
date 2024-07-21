# insurance-churn

This repo is for insurance churn prediction model.

## Folder Structure

```
project/
│
├── src/
│   ├── data_cleaning.py
│   ├── feature_generation.py
│   ├── handling_missing_values.py
│   ├── handling_outliers.py
│   └── models_training/
│       ├── logistic_regression.py
│       ├── random_forest.py
│       └── xgboost_model.py
│
├── runner/
│   ├── train_model.py
│   └── inference.py
│
├── config/
│   ├── config.json
│   └── config.py
|
├── models/
│   └── (Generated models will be saved here)
|
├── data/
│   └── (Data will be saved here)
│
├── templates
│   └── index.html
├── static
│   └── styles.css
|
│
├── workflow.py (Trigger Training and Inference run passed as parameter)
│
├── app/
│   └── app.py
│
├── Dockerfile
├── Makefile
├── Readme.md
└── requirements.txt
```

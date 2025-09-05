#!/usr/bin/env python3
"""
eval_algorithms.py

Train the four algorithms on 123.csv (median impute + optional scaling for SVC)
and print accuracy and ROC AUC for each model.

Run: python eval_algorithms.py
"""
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

DATA = os.path.join(os.path.dirname(__file__), '123.csv')

def load_data(path=DATA):
    df = pd.read_csv(path)
    if 'status' not in df.columns:
        raise RuntimeError('No status column found in CSV')
    X = df.drop(columns=['url', 'status'], errors='ignore')
    X = X.apply(pd.to_numeric, errors='coerce')
    y = df['status']
    return X, y

def build_models():
    models = {
        'dt': DecisionTreeClassifier(random_state=42),
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'et': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'svc': Pipeline([('scaler', StandardScaler()), ('svc', SVC(probability=True, kernel='rbf'))])
    }
    return models

def evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, m in models.items():
        try:
            m.fit(X_train, y_train)
        except Exception as e:
            print(f'Error training {name}:', e)
            results[name] = {'accuracy': None, 'auc': None}
            continue
        preds = m.predict(X_test)
        acc = accuracy_score(y_test, preds)
        auc = None
        try:
            if hasattr(m, 'predict_proba'):
                probs = m.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, probs)
            else:
                # try decision_function
                scores = m.decision_function(X_test)
                auc = roc_auc_score(y_test, scores)
        except Exception:
            auc = None
        results[name] = {'accuracy': acc, 'auc': auc}
    return results

def print_table(results):
    print('\nModel		Accuracy	ROC AUC')
    print('-----------------------------------------')
    for name, r in results.items():
        acc = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else 'N/A'
        auc = f"{r['auc']:.4f}" if r['auc'] is not None else 'N/A'
        print(f"{name:<8}\t{acc}\t\t{auc}")

def main():
    if not os.path.exists(DATA):
        print('Dataset not found at', DATA)
        sys.exit(1)
    X, y = load_data()
    feature_columns = list(X.columns)
    imputer = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)
    X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42, stratify=y)
    models = build_models()
    results = evaluate(models, X_train, X_test, y_train, y_test)
    print_table(results)

if __name__ == '__main__':
    main()

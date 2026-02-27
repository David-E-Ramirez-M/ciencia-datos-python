# Clasificacion End-to-End

- Dataset: `demo_breast_cancer`
- Mejor modelo por ROC-AUC CV: **logistic_regression**
- ROC-AUC holdout: **0.9954**
- Average Precision holdout: **0.9971**

## Classification Report (holdout)
```text
              precision    recall  f1-score   support

           0       0.91      0.98      0.94        42
           1       0.99      0.94      0.96        72

    accuracy                           0.96       114
   macro avg       0.95      0.96      0.95       114
weighted avg       0.96      0.96      0.96       114

```

Confusion matrix: `reports\figures\cm_demo_breast_cancer_logistic_regression.png`
Tabla comparativa CV: `reports\demo_breast_cancer_model_comparison.csv`
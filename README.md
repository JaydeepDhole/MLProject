# ML Project

## a. Problem statement
Build and compare multiple Machine Learning classification models on a suitable classification dataset (≥500 instances and ≥12 features).  
Evaluate each model using the required metrics and provide a Streamlit-based UI for inference and evaluation.

---

## b. Dataset description  
**Dataset Name:** Breast Cancer Wisconsin (Diagnostic)  
**Type:** Binary classification  
**Instances:** 569  
**Features:** 30 numeric features  
**Target:** `target` (0 = malignant, 1 = benign)

This dataset satisfies assignment constraints:
- Instances ≥ 500 ✅
- Features ≥ 12 ✅

---

## c. Models used  
(1 mark for all metrics for each model)

The following models were trained on the same dataset and evaluated on the hold-out test split:

- Logistic Regression  
- Decision Tree  
- kNN  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)

### Comparison Table (Evaluation Metrics)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9737 | 0.9884 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest (Ensemble) | 0.9561 | 0.9931 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| XGBoost (Ensemble) | 0.9561 | 0.9954 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |

---

### Observations on the performance of each model on the chosen dataset  

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall performance (highest accuracy and MCC, very high AUC). Indicates dataset is highly separable and LR generalizes well. |
| Decision Tree | Lowest accuracy/AUC/MCC among models. Likely overfits/underfits depending on splits; high precision but lower recall indicates more false negatives vs others. |
| kNN | Excellent recall (1.0) meaning it catches all positives in test set; very strong accuracy and MCC. Sensitive to scaling but performs very well here. |
| Naive Bayes | Good performance with high AUC, but slightly lower accuracy/MCC than LR/kNN due to independence assumption limitations. Very fast and simple baseline. |
| Random Forest (Ensemble) | Strong and stable performance (high AUC and good MCC). Improves over single tree by reducing variance; good balance of precision/recall. |
| XGBoost (Ensemble) | Very high AUC (ties best) and high recall. Similar accuracy to Random Forest but slightly different precision/recall tradeoff; strong non-linear learner. |





## Step to Run program
```bash
python model\model_utils.py --force
python model\train_models.py --force
streamlit run app.py

```


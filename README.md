# Customer Churn Prediction Pipeline (ML Benchmarking & Keras Neural Network)

An end-to-end supervised machine learning and deep learning classification pipeline built to predict telecommunications customer attrition across **7,032 IBM Telco customer records**, combining class-imbalance mitigation (**SMOTE**), ensemble tree benchmarking, and a regularized **TensorFlow/Keras Artificial Neural Network (ANN)**.

## System Architecture Flow

```text
IBM Telco Dataset (7,043 rows)
        │
        ▼
Data Cleaning (Drop customerID, coerce TotalCharges, drop NaNs -> 7,032 rows)
        │
        ▼
One-Hot Encoding (pd.get_dummies) ──▶ Stratified 80/20 Train-Test Split
                                              │
                                              ▼
                                      StandardScaler Fit/Transform
                                              │
                                              ▼
                                 SMOTE Oversampling (Train Set Only)
                                              │
                   ┌──────────────────────────┴──────────────────────────┐
                   ▼                                                     ▼
     Supervised ML Benchmarking                          TensorFlow / Keras Sequential ANN
(Logistic Regression, RF, GBM, XGBoost)               (Dense 32 -> Dropout 0.3 -> Dense 16 -> Sigmoid)
                   │                                                     │
                   └──────────────────────────┬──────────────────────────┘
                                              ▼
                       ROC-AUC Curve Comparison & Confusion Matrix
```

## Key Technical Highlights
- **Data Cleaning & Leakage Prevention:** Resolved blank-string anomalies in `TotalCharges` (`tenure == 0`) prior to scaling, and executed stratified train-test splitting **before** applying `SMOTE` oversampling strictly to the training set to prevent data leakage.
- **Multi-Model Benchmarking:** Evaluated linear, bagging, boosting, and deep neural architectures on the same scaled test split using **ROC-AUC** rather than raw accuracy alone to account for the ~73.5% vs. 26.5% class skew.
- **Regularized Deep Learning Classifier:** Implemented a multi-layer Keras Sequential ANN (`Dense` + `Dropout(0.3)` / `Dropout(0.2)`) compiled with Adam and binary cross-entropy loss.

## Sample Execution Logs & Metrics

### Class Balancing via SMOTE (Training Split)
```text
Before SMOTE: {0: 4130, 1: 1495}
After SMOTE:  {0: 4130, 1: 4130}
```

### Model Performance Comparison (Test ROC-AUC)
| Model Architecture | Framework | Evaluation Metric (ROC-AUC) |
| :--- | :--- | :---: |
| **Logistic Regression** | Scikit-Learn | **~0.833+** |
| **Gradient Boosting Classifier** | Scikit-Learn | **~0.834+** |
| **XGBoost Classifier** | XGBoost | **~0.834+** |
| **Random Forest Classifier** | Scikit-Learn | **~0.835+** |
| **Sequential ANN (Dense + Dropout)** | TensorFlow / Keras | **~0.830+** |

*(Exact values are printed and plotted in the ROC-AUC comparison chart inside `customer_churn_ml_ann_pipeline.ipynb`.)*

## Tech Stack
- **Data Manipulation & Visualization:** Python, NumPy, Pandas, Matplotlib, Seaborn
- **Machine Learning & Imbalance Handling:** Scikit-Learn, XGBoost, Imbalanced-Learn (`SMOTE`)
- **Deep Learning:** TensorFlow, Keras (`Sequential`, `Dense`, `Dropout`)

## Setup & Execution
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Open `customer_churn_ml_ann_pipeline.ipynb` in Jupyter Notebook or Google Colab and run all cells.

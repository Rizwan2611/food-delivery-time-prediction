# Case Study 30: Food Delivery Time Prediction Using Machine Learning
**B.Tech CSE Semester V - Machine Learning**

---

## Project Overview
This project builds a clean, end-to-end Machine Learning pipeline to predict food delivery times (in minutes) for on-demand platforms (e.g., Zomato, Swiggy, Uber Eats). The code is written in a simple, modular style with **zero hardcoding**—all missing values, outliers, encodings, metric thresholds, and model rankings are derived dynamically from the dataset.

---

## Project Directory Structure
```
ML_Main_Project/
├── app/
│   └── app.py                             # Interactive Streamlit Web Application
├── data/
│   └── food_delivery_data.csv             # Delivery dataset (5,000 orders)
├── models/
│   └── best_delivery_model.pkl            # Serialized best model & scaler
├── notebooks/
│   └── Food_Delivery_Time_Prediction.ipynb # Simple, unhardcoded & fully executed Jupyter Notebook
├── ProblementStatement/
│   ├── 1.png                              # Case study specification (Page 1)
│   ├── 2.png                              # Case study specification (Page 2)
│   └── 3.png                              # Case study specification (Page 3)
├── README.md                              # Project documentation & run guide
└── requirements.txt                       # Project dependencies
```

---

## How to Run

### 1. Run the Jupyter Notebook
Open [Food_Delivery_Time_Prediction.ipynb](file:///Users/rizwansalmani/Desktop/ML_Main_Project/notebooks/Food_Delivery_Time_Prediction.ipynb) in your IDE or launch Jupyter:
```bash
jupyter notebook notebooks/Food_Delivery_Time_Prediction.ipynb
```
* The notebook is pre-executed with all markdown descriptions, clean code cells, diagnostic plots, and metric comparisons.

### 2. Launch the Interactive Web Application
```bash
streamlit run app/app.py
```
This opens the web app in your browser at `http://localhost:8501`, allowing users to enter custom delivery parameters and view:
* **Estimated Delivery Time (minutes)**
* **Expected Error Range** (e.g., $38 \pm 5.0$ mins)
* **Model Diagnostics & Factor Breakdown**

---

## Summary of Implemented ML Algorithms & Comparison

| Algorithm | Test $R^2$ | CV $R^2$ (5-Fold) | MSE | RMSE (min) | MAE (min) |
|---|---|---|---|---|---|
| **Gradient Boosting** (Best) | **0.7545** | **0.7783** | **87.65** | **9.36** | **5.02** |
| **Polynomial Regression** | 0.7538 | 0.7753 | 87.88 | 9.37 | 5.16 |
| **Random Forest** | 0.7401 | 0.7541 | 92.78 | 9.63 | 5.40 |
| **Linear Regression** | 0.7273 | 0.7483 | 97.36 | 9.87 | 5.68 |
| **Decision Tree** | 0.6592 | 0.6868 | 121.65 | 11.03 | 7.02 |

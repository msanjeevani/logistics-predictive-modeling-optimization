# Logistics Predictive Modeling and Optimization

A Python-based logistics analytics project that predicts delivery times using
Random Forest Regression and presents model evaluation results through a Flask
web dashboard.

## Project Objectives

- Predict delivery time using logistics-related features.
- Evaluate model performance using RMSE, MAE, and R².
- Tune model hyperparameters using cross-validation.
- Analyze feature importance.
- Present prediction outputs and visualizations in a web dashboard.
- Recommend data-informed logistics improvement strategies.

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Flask
- HTML and CSS

## Project Structure

```text
logistics-advanced-data-analysis/
├── predictive_modeling.py
├── app.py
├── requirements.txt
├── README.md
└── output/
    ├── model_evaluation_metrics.csv
    ├── delivery_time_predictions.csv
    ├── feature_importance.csv
    ├── feature_importance.png
    └── actual_vs_predicted.png
```

## Setup

Python 3.10 or newer is recommended.

### 1. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run predictive modeling

```bash
python predictive_modeling.py
```

This creates evaluation metrics, test predictions, feature importance data,
and charts in the `output/` folder.

### 4. Start the dashboard

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

You can also use the dashboard button to rerun the analysis.

## Dataset

The model expects these columns in:

`output/cleaned_logistics_data.csv`

- `distance_km`
- `package_weight_kg`
- `fuel_cost_usd`
- `traffic_index`
- `delivery_time_minutes`

The first four columns are model features. `delivery_time_minutes` is the
target to predict.

If the CSV does not exist, the modeling script creates simulated demonstration
data. Make sure to distinguish simulated results from real-world findings.

## Evaluation Metrics

- **RMSE:** Penalizes larger prediction errors and is measured in minutes.
- **MAE:** Average absolute prediction error, measured in minutes.
- **R²:** Indicates how much target variation is explained by the model.

Cross-validation and grid search are used to help assess and tune the model.

## Generated Outputs

- `model_evaluation_metrics.csv`
- `delivery_time_predictions.csv`
- `feature_importance.csv`
- `feature_importance.png`
- `actual_vs_predicted.png`

## Optimization Recommendations

The analysis may help logistics teams investigate traffic-aware scheduling,
delivery-time planning, route performance, and resource allocation. These are
recommendations for further evaluation; the model does not automatically solve
a route optimization problem.

## Limitations

Model performance depends on data quality and how representative the dataset
is. Simulated data is for demonstrating the workflow only. Validate results
using suitable real-world data before making operational decisions.

## GitHub

https://github.com/msanjeevani/logistics-advanced-data-analysis

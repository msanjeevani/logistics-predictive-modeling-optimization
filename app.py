
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, render_template_string, send_from_directory, redirect, url_for, flash

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "local-development-key")

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Logistics Predictive Analytics</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0; font-family: Arial, sans-serif;
            background: #f3f6fb; color: #182230;
        }
        header {
            background: #142d4e; color: white;
            padding: 28px 6%;
        }
        header h1 { margin: 0 0 8px; }
        header p { margin: 0; color: #d9e5f5; }
        main { max-width: 1150px; margin: 28px auto; padding: 0 18px; }
        .actions { margin-bottom: 22px; }
        button, .download {
            display: inline-block; padding: 11px 16px;
            border: 0; border-radius: 7px; cursor: pointer;
            background: #1769aa; color: white; text-decoration: none;
            font-weight: bold; margin: 4px;
        }
        .notice {
            background: #fff4d6; padding: 12px;
            border-radius: 8px; margin-bottom: 18px;
        }
        .cards {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
            gap: 16px; margin: 20px 0;
        }
        .card, .panel {
            background: white; padding: 20px; border-radius: 12px;
            box-shadow: 0 3px 12px #142d4e12;
        }
        .card h3 { font-size: 14px; color: #53657a; margin-top: 0; }
        .value { font-size: 27px; font-weight: bold; color: #1769aa; }
        .panel { margin: 18px 0; overflow-wrap: anywhere; }
        .charts {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 18px;
        }
        .charts img { width: 100%; height: auto; border-radius: 7px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { text-align: left; padding: 10px; border-bottom: 1px solid #e4eaf1; }
        th { background: #eef4fa; }
        footer { text-align: center; color: #64748b; padding: 25px; }
    </style>
</head>
<body>
<header>
    <h1>Logistics Predictive Analytics</h1>
    <p>Delivery Time Prediction | Random Forest Regression | Model Evaluation</p>
</header>
<main>
    {% for message in messages %}
      <div class="notice">{{ message }}</div>
    {% endfor %}

    <div class="actions">
        <form method="post" action="{{ url_for('run_model') }}">
            <button type="submit">Run / Refresh Model Analysis</button>
        </form>
        <a class="download" href="{{ url_for('download_file', filename='model_evaluation_metrics.csv') }}">Download Metrics</a>
        <a class="download" href="{{ url_for('download_file', filename='delivery_time_predictions.csv') }}">Download Predictions</a>
        <a class="download" href="{{ url_for('download_file', filename='feature_importance.csv') }}">Download Feature Importance</a>
    </div>

    {% if not has_metrics %}
      <div class="notice">
        No evaluation results found yet. Click “Run / Refresh Model Analysis” to run
        predictive_modeling.py and generate the outputs.
      </div>
    {% endif %}

    <div class="cards">
        <div class="card">
            <h3>RMSE (minutes)</h3>
            <div class="value">{{ metrics.get('RMSE_minutes', '—') }}</div>
        </div>
        <div class="card">
            <h3>MAE (minutes)</h3>
            <div class="value">{{ metrics.get('MAE_minutes', '—') }}</div>
        </div>
        <div class="card">
            <h3>R² Score</h3>
            <div class="value">{{ metrics.get('R2_Score', '—') }}</div>
        </div>
        <div class="card">
            <h3>Cross-validation MAE (minutes)</h3>
            <div class="value">{{ metrics.get('Mean_CV_MAE_minutes', '—') }}</div>
        </div>
    </div>

    <section class="panel">
        <h2>Model Evaluation Metrics</h2>
        {% if metric_rows %}
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            {% for row in metric_rows %}
            <tr><td>{{ row.Metric }}</td><td>{{ row.Value }}</td></tr>
            {% endfor %}
        </table>
        {% else %}
        <p>Metrics will appear here after the analysis script runs.</p>
        {% endif %}
    </section>

    <section class="panel">
        <h2>Model Visualizations</h2>
        <div class="charts">
            {% for chart in charts %}
            <div>
                <h3>{{ chart.title }}</h3>
                <img src="{{ url_for('output_chart', filename=chart.filename) }}"
                     alt="{{ chart.title }}">
            </div>
            {% endfor %}
        </div>
        {% if not charts %}
        <p>Charts will appear after the model has run successfully.</p>
        {% endif %}
    </section>

    <section class="panel">
        <h2>Feature Importance</h2>
        {% if importance_rows %}
        <table>
            <tr><th>Feature</th><th>Importance</th></tr>
            {% for row in importance_rows %}
            <tr><td>{{ row.Feature }}</td><td>{{ row.Importance }}</td></tr>
            {% endfor %}
        </table>
        {% else %}
        <p>Feature importance results are not available yet.</p>
        {% endif %}
    </section>

    <section class="panel">
        <h2>Optimization Recommendations</h2>
        <ul>
            <li>Use predicted delivery times to plan realistic delivery windows.</li>
            <li>Review high-traffic deliveries for possible scheduling or route changes.</li>
            <li>Compare route and vehicle performance before reallocating resources.</li>
            <li>Monitor prediction errors and retrain with updated real-world data.</li>
        </ul>
        <p><strong>Note:</strong> These are recommendations, not automatically optimized routes.</p>
    </section>
</main>
<footer>Week 4 Project | Predictive Modeling and Optimization in Logistics Systems</footer>
</body>
</html>
"""


def read_csv_safe(filename):
    path = OUTPUT_DIR / filename
    if path.exists():
        try:
            return pd.read_csv(path)
        except (pd.errors.ParserError, OSError, UnicodeDecodeError):
            return pd.DataFrame()
    return pd.DataFrame()


@app.route("/")
def index():
    messages = []

    metrics_df = read_csv_safe("model_evaluation_metrics.csv")
    metrics = {}

    if not metrics_df.empty and {"Metric", "Value"}.issubset(metrics_df.columns):
        for _, row in metrics_df.iterrows():
            try:
                metrics[str(row["Metric"])] = f"{float(row['Value']):.3f}"
            except (TypeError, ValueError):
                metrics[str(row["Metric"])] = str(row["Value"])

    importance_df = read_csv_safe("feature_importance.csv")
    importance_rows = (
        importance_df.to_dict(orient="records")
        if {"Feature", "Importance"}.issubset(importance_df.columns)
        else []
    )

    chart_specs = [
        ("feature_importance.png", "Feature Importance"),
        ("actual_vs_predicted.png", "Actual vs Predicted Delivery Time"),
    ]
    charts = [
        {"filename": filename, "title": title}
        for filename, title in chart_specs
        if (OUTPUT_DIR / filename).is_file()
    ]

    metric_rows = (
        metrics_df.to_dict(orient="records")
        if {"Metric", "Value"}.issubset(metrics_df.columns)
        else []
    )

    return render_template_string(
        HTML,
        metrics=metrics,
        metric_rows=metric_rows,
        importance_rows=importance_rows,
        charts=charts,
        has_metrics=not metrics_df.empty,
        messages=messages,
    )


@app.route("/run-model", methods=["POST"])
def run_model():
    script_path = BASE_DIR / "predictive_modeling.py"

    if not script_path.is_file():
        flash("predictive_modeling.py was not found. Place it beside app.py.")
        return redirect(url_for("index"))

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        if result.returncode == 0:
            flash("Model analysis completed. Dashboard outputs have been refreshed.")
        else:
            error_text = (result.stderr or result.stdout or "Unknown error")[-1200:]
            flash("Model execution failed: " + error_text)
    except subprocess.TimeoutExpired:
        flash("Model execution timed out. Try running predictive_modeling.py in the terminal.")
    except OSError as exc:
        flash(f"Could not start the model script: {exc}")

    return redirect(url_for("index"))


@app.route("/output-chart/<path:filename>")
def output_chart(filename):
    allowed = {"feature_importance.png", "actual_vs_predicted.png"}
    if filename not in allowed:
        return "Chart not found", 404
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/download/<path:filename>")
def download_file(filename):
    allowed = {
        "model_evaluation_metrics.csv",
        "delivery_time_predictions.csv",
        "feature_importance.csv",
    }
    if filename not in allowed:
        return "File not found", 404

    file_path = OUTPUT_DIR / filename
    if not file_path.is_file():
        flash(f"{filename} is not available yet. Run the model first.")
        return redirect(url_for("index"))

    return send_from_directory(
        str(OUTPUT_DIR),
        filename,
        as_attachment=True,
    )


if __name__ == "__main__":
    app.run(debug=True)

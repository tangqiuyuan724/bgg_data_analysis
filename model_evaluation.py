import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error, r2_score

from feature_engineering import X_test

sns.set(style="whitegrid")

# 1. loading data and model
print("loading trained pipeline and test data")
try:
    with open("output_set/processed_data.pkl", "rb") as f:
        data = pickle.load(f)
        X_test = data["X_test"]
        y_test = data["y_test"]
        feature_names = data["feature_names"]

    with open("output_set/trained_pipeline.pkl", "rb") as f:
        pipeline = pickle.load(f)

    print("loaded trained pipeline and test data")
    print("-"*30)
except FileNotFoundError:
    raise SystemExit("No trained pipeline available")

# 2. evaluation
results = {}
predictions = {}

for name,pipe in pipeline.items():
    # prediction: pipeline will automatically execute scaler.transform and predict
    y_pred = pipe.predict(X_test)
    predictions[name] = y_pred

    # calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[name] = {"RMSE": rmse, "R2": r2}
    print(f"[{name}] prediction done -> RMSE = {rmse:.4f} | R2 = {r2:.4f}")

print("-"*30)

# 3. visualization of results
results_df = pd.DataFrame(results).T
# A. RMSE
plt.figure(figsize=(10, 5))
ax = sns.barplot(x=results_df.index, y=results_df['RMSE'], palette='viridis')
plt.title('Final Model Evaluation: RMSE (Lower is Better)', fontsize=14)
plt.ylabel('Root Mean Squared Error (Points)')
plt.ylim(0.25, 0.40)

for i, v in enumerate(results_df['RMSE']):
    ax.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# B. feature importances in random forest
rf_pipe = pipeline['Random Forest']
rf_model = rf_pipe.named_steps['regressor']
importances = rf_model.feature_importances_
# get first fifteen
indices = np.argsort(importances)[-15:]

plt.figure(figsize=(12, 8))
plt.title('What drives a Board Game Rating? (Random Forest Insights)', fontsize=15)
plt.barh(range(len(indices)), importances[indices], align='center', color='#2ecc71')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Relative Importance')
plt.tight_layout()
plt.show()

# C. truth vs prediction
plt.figure(figsize=(8, 8))
plt.scatter(y_test, predictions["Random Forest"], alpha=0.3, s=15, color='#3498db', label='Predictions')
# drawing the perfect diagonal
plt.plot([5, 8.5], [5, 8.5], 'r--', lw=2, label='Perfect Fit')
plt.xlabel('Actual Bayes Score')
plt.ylabel('Predicted Bayes Score')
plt.title('Prediction Accuracy: Random Forest', fontsize=15)
plt.legend()
plt.show()

print("-"*30)
print("final results:")
print(results_df)
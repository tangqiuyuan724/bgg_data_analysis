import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score

from feature_engineering import X_train

sns.set(style="whitegrid")

# 1. loading data
print("loading data")
try:
    with open('output_set/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
        X_train= data['X_train']
        X_test= data['X_test']
        y_train = data['y_train']
        y_test= data['y_test']
        scaler = data['scaler']
        feature_names = data['feature_names']
    print("data loaded")
except FileNotFoundError:
    raise SystemExit("processed_data.pkl not found")

# 2. Gradient Boosting + automatic tuning
print("-"*30)
print("start hyperparameter searching")

# define basic model
gb_model = GradientBoostingRegressor(random_state=42)

# define search scope
param_dist = {
    'n_estimators': [100,200,300],  # number of tree
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 1.0]
}

# random search
search = RandomizedSearchCV(
    estimator=gb_model,
    param_distributions=param_dist,
    n_iter=10, # randomly try 10 combinations
    scoring='neg_root_mean_squared_error', # aim to optimize rmse
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# start training
search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"best params combination: {search.best_params_}")
print("-"*30)

# 3. global evaluation
y_pred = best_model.predict(X_test)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
final_r2 = r2_score(y_test, y_pred)

print(f"final rmse: {final_rmse:.4f}")
print(f"final r2: {final_r2:.4f}")
print("-"*30)

# 4. deep dive analysis
print("start multidimensional slice analysis")
# A. data restoration
# convert the normalized values back to the original values
# X_test_restored = scaler.inverse_transform(X_test)
# df_analysis = pd.DataFrame(X_test_restored, columns=feature_names)
df_analysis = pd.DataFrame(X_test, columns=feature_names)

# contact real values, prediction values and errors into one table
df_analysis['Actual'] = y_test.values
df_analysis['Predicted'] = y_pred
df_analysis['Error'] = df_analysis['Actual'] - df_analysis['Predicted']
df_analysis['Abs_Error'] = df_analysis['Error'].abs()

# pic1: prediction errors in different complexity
df_analysis['Complexity_Group'] = pd.cut(
    df_analysis['Complexity Average'],
    bins=[0, 2, 3.5, 5],
    labels=['Light (0-2)', 'Medium (2-3.5)', 'Heavy (3.5-5)']
)
rmse_by_complexity = df_analysis.groupby('Complexity_Group')['Error'].apply(lambda x: np.sqrt((x**2).mean()))

plt.figure(figsize=(8, 5))
sns.barplot(x=rmse_by_complexity.index, y=rmse_by_complexity.values, palette='magma')
plt.title('Performance Check: RMSE by Complexity', fontsize=14)
plt.ylabel('RMSE (Error)')
plt.ylim(0.2, 0.45)
for i, v in enumerate(rmse_by_complexity):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# pic2: prediction errors in different year published
df_analysis['Year_Group'] = pd.cut(
    df_analysis['Year Published'],
    bins=[1900, 2000, 2010, 2015, 2025],
    labels=['Pre-2000', '2000-2010', '2010-2015', 'Post-2015']
)
rmse_by_year = df_analysis.groupby('Year_Group')['Error'].apply(lambda x: np.sqrt((x**2).mean()))

plt.figure(figsize=(8, 5))
sns.barplot(x=rmse_by_year.index, y=rmse_by_year.values, palette='viridis')
plt.title('Performance Check: RMSE by Era', fontsize=14)
plt.ylabel('RMSE (Error)')
plt.ylim(0.2, 0.4)
for i, v in enumerate(rmse_by_year):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

# pic3: residual plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_analysis['Predicted'], y=df_analysis['Error'], alpha=0.3, color='#8e44ad')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residual Plot: Is the error random?', fontsize=14)
plt.xlabel('Predicted Score')
plt.ylabel('Residual (Actual - Predicted)')
plt.tight_layout()
plt.show()
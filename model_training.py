import pickle
import time
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from feature_engineering import X_train

# 1. loading data
print("loading data")
with open("output_set/processed_data.pkl", "rb") as f:
    data = pickle.load(f)
    X_train = data["X_train"]
    y_train = data["y_train"]
    print(f'samples volume: {len(X_train)}')

# 2. define pipeline
print('constructing pipeline')
pipeline = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ]),
    "Random Forest": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42))
    ]),
    "Neural Network": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", MLPRegressor(hidden_layer_sizes=(64,32), max_iter=500, random_state=42))
    ])
}

# 3. fitting/training
trained_pipeline = {}

print("-"*30)
for name, pipe in pipeline.items():
    start = time.time()
    print(f"start training pipeline: [{name}] ... ")
    # pipe.fit() will automatically execute scaler.fit_transform(X) and regressor.fit(X_scaled, y)
    pipe.fit(X_train, y_train)

    end = time.time()
    duration = end - start
    print(f"done training pipeline! duration: {duration:.2f} seconds")
    trained_pipeline[name] = pipe

# 4. save pipeline
print("-"*30)
print("saving trained pipeline")
with open("output_set/trained_pipeline.pkl", "wb") as f:
    pickle.dump(trained_pipeline, f)

print("saved trained pipeline in trained_pipeline.pkl")
print("-"*30)
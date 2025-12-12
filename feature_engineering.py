import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# loading final cleaned output_set
df = pd.read_csv('output_set/final_cleaned_bgg_data.csv')

# 1. constructing features
print("constructing features...")

# A. Mechanics: extract top 20 hottest mechanics
df['Mechanics_List'] = df['Mechanics'].apply(lambda x: [d.strip() for d in x.split(',')])
all_mechanics = [m for sublist in df['Mechanics_List'] for m in sublist if m!= 'Unknown']
top_20_mechanics = [m[0] for m in Counter(all_mechanics).most_common(20)]
for mech in top_20_mechanics:
    # cleaning blank spaces and special characters in column names to avoid errors
    safe_mech_name = mech.replace(' ', '').replace('/','_')
    df[f'Mech_{safe_mech_name}'] = df['Mechanics_List'].apply(lambda x: 1 if mech in x else 0)

# B. Domains: extract top 5
df['Domains_List'] = df['Domains'].apply(lambda x: [d.strip() for d in x.split(',')])
all_domains = [d for sublist in df['Domains_List'] for d in sublist if d!= 'Unknown']
top_domains = [d[0] for d in Counter(all_domains).most_common(5)]
for dom in top_domains:
    safe_dom_name = dom.replace(' ', '').replace("'","")
    df[f'Dom_{safe_dom_name}'] = df['Domains_List'].apply(lambda x: 1 if dom in x else 0)

# C. Play time: Binning
# solve non-linear problem: turn numerical value into labels
bins = [0,30,60,120,240,999]
labels = ['Time_0_30', 'Time_30_60', 'Time_60_120', 'Time_120_240', 'Time_GT_240']
df['Time_Category'] = pd.cut(df['Play Time'], bins=bins, labels=labels)

# one-hot encoding: turn labels into 0/1 columns
time_dummies = pd.get_dummies(df['Time_Category'], drop_first=False).astype(int)
df = pd.concat([df, time_dummies], axis=1)

# 2. define final features X and target Y
# numerical features
num_features = ['Year Published', 'Min Players', 'Max Players', 'Min Age', 'Complexity Average', 'Play Time']
# mechanics features (with prefix 'Mech_')
mech_features = [col for col in df.columns if col.startswith('Mech_')]
# domains features (with prefix 'Dom_')
dom_features = [col for col in df.columns if col.startswith('Dom_')]
# play time bins features
time_features = labels

# merge all features names
features_cols = num_features + mech_features + dom_features+ time_features
print(f"final chosen {len(features_cols)} features:\n{features_cols}")

X = df[features_cols]
y = df['bayesaverage']

# 3. output_set split and standardization
# split training and testing output_set (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# save processed data for model
data_to_save = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'scaler': StandardScaler(),
    'feature_names': features_cols
}
with open('output_set/processed_data.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("-" * 30)
print("feature engineering completes! data saved in processed_data.pkl")
print(f"shape of training set: {X_train.shape}")
print(f"shape of test set: {X_test.shape}")
print("-" * 30)
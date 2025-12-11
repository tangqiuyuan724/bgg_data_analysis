import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# loading cleaned dataset
df = pd.read_csv('dataset/cleaned_bgg_data.csv')

sns.set(style='whitegrid')

# statistics of dataset
print("info:")
print(df.info())
print("statistics:")
print(df.describe())

# 1. target value analysis: distribution of bayes average
plt.figure(figsize=(10,6))
sns.histplot(df['bayesaverage'], bins=50, kde=True, color='#3498db')
plt.title('Target Distribution: Bayes Average Score', fontsize=15)
plt.xlabel('Score')
plt.axvline(df['bayesaverage'].mean(), color='r', linestyle='--', label=f"Mean: {df['bayesaverage'].mean():.2f}")
plt.legend()
plt.show()

# 2. key relation investigation: complexity vs ratings
plt.figure(figsize=(10, 6))
plt.hexbin(df['Complexity Average'], df['bayesaverage'], gridsize=30, cmap='Blues', mincnt=1)
plt.colorbar(label='Count of Games')
plt.title('Does Complexity drive higher Ratings?', fontsize=15)
plt.xlabel('Complexity (1=Light, 5=Heavy)')
plt.ylabel('Bayes Average')
sns.regplot(x='Complexity Average', y='bayesaverage', data=df, scatter=False, color='red', line_kws={"linewidth": 2})
plt.show()

# 3. year trend: year published vs ratings
plt.figure(figsize=(12, 6))
# filter out a very small number of games that are too early to see more clearly
df_recent = df[df['Year Published'] >= 1980]
sns.lineplot(data=df_recent, x='Year Published', y='bayesaverage', errorbar=None, color='green')
plt.title('Trend: Are modern games getting better scores?', fontsize=15)
plt.ylabel('Average Bayes Score')
plt.show()

# check the average of users rated after 2023
recent_stats = df[df['Year Published'] >= 2023].groupby('Year Published')['Users Rated'].mean()
print(f"mean of users rated after 2023: {recent_stats}")
# for model stability, exclude data in latest 1-2 years
cutoff_year = 2023
df_final_clean = df[df['Year Published'] <= cutoff_year].copy()
print(f"exclude data after {cutoff_year} to eliminate tail noise caused by Bayesian penalty")
print(f"current data volume: {len(df_final_clean)}")
df = df_final_clean
df.to_csv("dataset/final_cleaned_bgg_data.csv", index=False)

# year trend after cutting off
plt.figure(figsize=(12, 6))
df_recent = df[df['Year Published'] >= 1980]
sns.lineplot(data=df_recent, x='Year Published', y='bayesaverage', errorbar=None, color='green')
plt.title('Trend: Are modern games getting better scores? (excluding 2023-2025)', fontsize=15)
plt.ylabel('Average Bayes Score')
plt.show()

# 4. Correlation heatmap (numerical characteristics)
corr_cols = ['bayesaverage', 'Complexity Average', 'Year Published',
             'Min Age', 'Play Time', 'Min Players', 'Max Players']
corr_matrix = df[corr_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap', fontsize=15)
plt.show()
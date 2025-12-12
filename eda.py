import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# loading cleaned output_set
df = pd.read_csv('output_set/cleaned_bgg_data.csv')

sns.set(style='whitegrid')

# statistics of output_set
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
df.to_csv("output_set/final_cleaned_bgg_data.csv", index=False)

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

# 5. mechanics vs bayes average
# As a board game could have multiple mechanics, we need to explode them.
print("-"*30)
print("analyzing mechanics...")
# change string 'A,B,C' into list ['A','B','C']
df['Mech_List'] = df['Mechanics'].astype(str).apply(lambda x: x.split(','))
# Explode: one line becomes multiple lines, one line for each mechanism
df_exploded = df.explode('Mech_List')
df_exploded['Mech_List'] = df_exploded['Mech_List'].apply(lambda x: x.strip())
# filter out Unknown
df_exploded = df_exploded[df_exploded['Mech_List']!='Unknown']
# Statistics: only consider mechanisms that have occurred at least 100 times (to avoid small sample bias)
mech_counts = df_exploded['Mech_List'].value_counts()
valid_mechs = mech_counts[mech_counts > 100].index
df_valid_mechs = df_exploded[df_exploded['Mech_List'].isin(valid_mechs)]
# calculate mean of bayes average score in each mechanics
mech_ratings = df_valid_mechs.groupby('Mech_List')['bayesaverage'].mean().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=mech_ratings.values, y=mech_ratings.index,hue=mech_ratings.index, palette='viridis',legend=False)
plt.title('Mechanics by Average Bayes Rating', fontsize=15)
plt.xlabel('Average Rating')
plt.axvline(df['bayesaverage'].mean(), color='red', linestyle='--', label='Global Average')
plt.legend()
plt.tight_layout()
plt.show()

# 6. domains comparison: strategic vs party vs children
print("-"*30)
print("analyzing domains...")
df['Dom_List'] = df['Domains'].astype(str).apply(lambda x: x.split(','))
df_dom_exploded = df.explode('Dom_List')
df_dom_exploded['Dom_List'] = df_dom_exploded['Dom_List'].apply(lambda x: x.strip())
# filter out Unknown
df_dom_exploded = df_dom_exploded[df_dom_exploded['Dom_List']!='Unknown']
# sort
dom_order = df_dom_exploded.groupby('Dom_List')['bayesaverage'].mean().sort_values(ascending=False).index
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_dom_exploded, x='bayesaverage', y='Dom_List',hue='Dom_List',legend=False, order=dom_order, palette='Set2')
plt.title('Rating Distribution by Domain', fontsize=15)
plt.xlabel('Bayes Average')
plt.ylabel('')
plt.tight_layout()
plt.show()

# 7. play time vs bayes average
print("-"*30)
print("analyzing play time...")
# creating duration bins
bins = [0,30,60,120,240,9999]
labels = ['< 30 min', '30-60 min', '1-2 hours', '2-4 hours', '> 4 hours']
df['Time_Category'] = pd.cut(df['Play Time'],bins=bins,labels=labels)
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='Time_Category', y='bayesaverage',palette='muted')
plt.title('Game Duration vs Rating', fontsize=15)
plt.xlabel('Play Time Category')
plt.ylabel('Bayes Average')
plt.tight_layout()
plt.show()
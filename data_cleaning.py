import pandas as pd
import numpy as np

# 1. loading dataset
print("loading data...")
# latin1(ISO-8859-1) encoding format has a higher fault tolerance
# many board games come from Europe especially German, thus including some special characters
df_bgg = pd.read_csv("dataset/BGG_Data_Set.csv", encoding="latin1")
df_ranks = pd.read_csv("dataset/boardgames_ranks.csv")
print("-"*30)
# check data info
print("information of BGG_Data_Set...")
print(df_bgg.head())
print(df_bgg.info())
print("information of boardgames_ranks...")
print(df_ranks.head())
print(df_ranks.info())
print("-"*30)
# 2. preprocessing
# delete null ID in df (16 rows in 20343 rows), transfer into type integer
df_bgg=df_bgg.dropna(subset=['ID'])
df_bgg['ID'] = df_bgg['ID'].astype(int)
print("after drop null id in BGG_Data_Set...")
print(df_bgg.info())
print("-"*30)

# 3. merging two dataset based on id
# using inner join to make sure only board games that exist in both csv will be retained
print("after merging data...")
df = pd.merge(df_bgg, df_ranks[['id', 'bayesaverage','yearpublished']], left_on='ID', right_on='id', how='inner')
print(df.head())
print(df.info())
print("-"*30)

# fixing year published value
def fix_year(row):
    """
    Cross-Reference
    if year published is invalid(<1900 or > 2025) in the left table, try to use year published in the right table
    """
    year_bgg = row['Year Published']
    year_rank = row['yearpublished']
    # check if year published in left table is invalid (<1900 or > 2025)
    if year_bgg <= 1900 or year_bgg > 2025:
        # check if year published in right table is valid
        if 1900 < year_rank <= 2025:
            return year_rank
    return year_bgg

# apply fix year
df['Year Published'] = df.apply(fix_year, axis=1)
# transfer 'Year Published' into int
df['Year Published'] = df['Year Published'].astype(int)
# drop duplicated column of 'id' and 'year published'
df = df.drop(columns=['yearpublished','id'],axis=1)

# 4. filtering and cleaning
# drop invalid values
before_count = len(df)
df = df[(df['Year Published']>1900) & (df['Year Published']<=2025)]
after_count = len(df)
print(f"with cross conference, we fix {before_count-after_count} rows which had invalid year published")

# Mechanics is a text feature, fill missing value with 'Unknown'
df['Mechanics'] = df['Mechanics'].fillna('Unknown')

# for key numerical features, if missing, delete the row directly, as it cannot be accurately filled
df = df.dropna(subset=['Complexity Average', 'Min Age', 'Play Time'])
print("-"*30)

# 5. check results
print(f"after cleaning, the final shape of dataset: {df.shape}")
print("info of dataset:")
print(df.info())
print("preview of the first five rows: ")
print(df[['ID', 'Name', 'Year Published', 'bayesaverage', 'Mechanics']].head())

# save cleaned data
df.to_csv("dataset/cleaned_bgg_data.csv", index=False)

# save the board games data which only exist in the right table
# these data lack same detail features, such as mechanics, complexity
print("-"*30)
bgg_ids = set(df_bgg['ID'])
ranks_ids = set(df_ranks['id'])

diff_ids = ranks_ids-bgg_ids
print(f"there are {len(ranks_ids)} rows in ranks table")
print(f"there are {len(bgg_ids)} rows in bgg table")
print(f"number of board games only existing in ranks table (missing detail features): {len(diff_ids)}")

df_missing_details = df_ranks[df_ranks['id'].isin(diff_ids)].copy()
df_missing_details.to_csv("dataset/games_without_details.csv", index=False)
print("preview of board games data without details:")
print(df_missing_details.head())
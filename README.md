to avoid data leakage, we will drop these columns before training:
* bgg rank/rank: since rank is decided by the bayes average, aka the target.
* rating average: it is a variant of bayes average, which is calculated from the average.
* users rated: when we predict a new game, it is impossible to know how many people will rate it.
* owned users: the same as users rated. the number is a result of market performance.
* name: it has little impact on prediction of bayes average.

cross conference to fix year published:
if year published is invalid(<1900 or > 2025) in the left table, try to use year published in the right table.
if both values of year published in two tables are invalid, drop the row.
we successfully fixed 238 rows which had invalid year published in this way.

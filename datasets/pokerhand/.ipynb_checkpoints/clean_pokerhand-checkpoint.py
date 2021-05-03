import pandas as pd

data = pd.read_csv("pokerhand-normalized.csv")

sList = list(map(lambda x: "s" + str(x), range(1,6)))

s = [pd.get_dummies(data[e], prefix=e) for e in sList]

cleaned = pd.concat([data] + s, axis=1)
cleaned.drop(sList, axis=1, inplace=True)

cleaned = cleaned.loc[:, ["class"] + list(filter(lambda x: x != "class", cleaned.columns))]

cleaned.to_csv("pokerhand-normalized_cleaned.csv", index=False)

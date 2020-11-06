# importing modules
import pandas as pd
import numpy as np

df_data = pd.read_csv("../../data/iris.data", header=None)  # getting data
df_data.columns = ["sepal_l", "sepal_w", "petal_l", "petal_w", "flower"]  # naming columns

# one hot encoding flower class
df_target = pd.get_dummies(df_data.flower, prefix="f")
df_data.drop("flower", axis=1, inplace=True) # deleting original flower variable

final_df = pd.concat([df_data, df_target], axis=1)
final_df = final_df.sample(frac=1).reset_index(drop=True)

df_data = final_df.iloc[:, [0, 1, 2, 3]].copy()
df_target = final_df.iloc[:, [4, 5, 6]].copy()

print(df_data.head())
print(df_target.head())

# saving dataframe
np.save("data", df_data.to_numpy())
np.save("target", df_target.to_numpy())
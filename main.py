"""
Project: Kaggle - Titanic Survival Prediction
Start: May 28th, 2024
Last touched: May 28th, 2024
Author: Madeleine L.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv("Data/train.csv")
# data["Survived"] = data["Survived"].astype(bool)
data["PassengerId"] = data["PassengerId"].astype(object)
print(data.head())
# print(data.dtypes)
# print(data.describe())

numerical_data = data.select_dtypes(include=["int", "float"])


# From simple correlation analysis it looks like fair and class influenced survival the most. The higher the fare cost
# the less likely you were to die, the lower the class (higher the numerical value) the lower the chance of survival. Of
# course these two variables are also highly correlated. As are parch and sibsp
correlation_matrix = numerical_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")


# 72.73% of children traveling with nannies survived vs 57% of children traveling with a family member
children_data = data[(data["Age"] < 16) & (data["Parch"] != 0)]
# print(sum(children_data["Survived"])/len(children_data["Survived"]))
print(children_data.describe())





# for column in data.columns:
#     if column not in ["PassengerId", "Survived", "Name"]:
#         if data[column].dtype in ["int64", "float64"]:
#             plt.figure()
#             plt.scatter(data["Survived"], data[column])
#             plt.xlabel("Survived")
#             plt.ylabel(column)
#             plt.title(f"Survived vs {column}")
#         else:
#             count_data = data.groupby(["Survived", column]).size().unstack()
#             count_data.plot(kind="bar", stacked=True)
#             plt.xlabel("Survived")
#             plt.ylabel(column)
#             plt.title(f"Survived vs {column}")

# plt.show()

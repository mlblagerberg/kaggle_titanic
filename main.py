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
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------- IMPORT DATA ------------------------------------ #
data = pd.read_csv("Data/train.csv")
# data["Survived"] = data["Survived"].astype(bool)
data["PassengerId"] = data["PassengerId"].astype(object)
data[["LastName", "SuffixFirstName"]] = data["Name"].str.split(",", expand=True)
data[["Suffix", "FirstName"]] = data["SuffixFirstName"].str.split(".", n=1, expand=True)
data.drop(columns=["Name", "SuffixFirstName"], inplace=True)
# print(data.head())
# print(data.columns)
# print(data.dtypes)
# print(data.describe())
# print(len(data[(data["Ticket"].isna())]))
# -------------------------------------- INITIAL ANALYSIS ------------------------------------ #
# From simple correlation analysis it looks like fair and class influenced survival the most. The higher the fare cost
# the less likely you were to die, the lower the class (higher the numerical value) the lower the chance of survival. Of
# course these two variables are also highly correlated. As are parch and sibsp
median_age = data["Age"].median()  # 28
# print(median_age)

# Correlation matrix
# numerical_data = data.select_dtypes(include=["int", "float"])
# correlation_matrix = numerical_data.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Matrix")
# plt.show()

# 72.73% of children traveling with nannies survived vs 57% of children traveling with a family member
children_data = data[(data["Age"] < 16) & (data["Parch"] != 0)]
# print(sum(children_data["Survived"])/len(children_data["Survived"]))
# print(children_data.describe())

# -------------------------------------- BASELINE MODEL ------------------------------------ #
# # Baseline only uses numerical features
base_data = data.dropna()
# features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
# X = (base_data[features]).values
# y = base_data["Survived"].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# # print("Shape of X_train:", X_train.shape)
#
model = LogisticRegression()
# model.fit(X_train, y_train)
#
# # prediction
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")  # Accuracy: 0.7027027027027027
#
# coefficients = model.coef_[0]
# for feature, coef in zip(features, coefficients):
#     print(f"Feature {feature}: {coef}")

# -------------------------------------- PREPROCESSING ------------------------------------ #
# columns_with_null = data.columns[data.isna().any()].tolist()
# # print(f"Columns with nulls: {columns_with_null}")  # Age, Cabin, Embarked
# # print(data[data["Cabin"].isna()].describe())
# data["Sex"] = data["Sex"].astype(bool)
# # print(data.dtypes)
# # print(data[["Cabin", "Pclass"]].head(50))
# # print(len(data["Ticket"].unique()))
# features = ["Pclass",
#             "Age",
#             "SibSp",
#             "Parch",
#             "Fare",
#             "Sex",
#             # "Embarked",
#             # "Ticket",
#             # "Cabin",
#             "Suffix"
#             ]
# X = (base_data[features]) #.values
# y = base_data["Survived"] #.values
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# preprocessor = ColumnTransformer(
#     transformers=[
#         # ('lifeboat', OneHotEncoder(), ["Sex"]),
#         # ('ice', OneHotEncoder(), ["Embarked"]),
#         # ('burg', OneHotEncoder(handle_unknown="ignore"), ["Ticket"]),
#         # ('white', OneHotEncoder(handle_unknown="ignore"), ["Cabin"]),
#         ('star', OneHotEncoder(handle_unknown="ignore"), ["Suffix"])
#     ])
#
# X_train_preprocessed = preprocessor.fit_transform(X_train)
# X_test_preprocessed = preprocessor.transform(X_test)
#
# model.fit(X_train_preprocessed, y_train)
#
# y_pred = model.predict(X_test_preprocessed)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")  # Accuracy: 0.6756756756756757
# # **************** After removing Embarked, Ticket, and Cabin Accuracy is 0.8108108108108109 **************** #
#
# coefficients = model.coef_[0]
# for feature, coef in zip(features, coefficients):
#     print(f"Feature {feature}: {coef}")
# print(data["Cabin"].unique())
# -------------------------------------- IMPUTE NULL AGE ------------------------------------ #
null_age = data[data["Age"].isna()]
null_age_has_family = null_age[(null_age["SibSp"] != 0) | (null_age["Parch"] != 0)]

# print(null_age.describe())
# print(null_age_with_family.describe())

null_age_with_family = pd.merge(data, null_age_has_family, on="LastName", how="inner")
# filtered_null_age_family = null_age_with_family[null_age_with_family["FirstName_x"] !=
#                                                 null_age_with_family["FirstName_y"]]
sorted_null_age_family = null_age_with_family.sort_values(by="LastName").drop_duplicates()
columns = ["PassengerId", "LastName", "Suffix_y", "FirstName_y", "Age_y", "Sex_y",
           "Suffix_x", "FirstName_x", "Age_x", "Sex_x"]
filtered_null_age = sorted_null_age_family[sorted_null_age_family.columns.intersection(columns)]
# sorted_null_age_family.drop(columns)
# print(sorted_null_age_family.columns)
# print(sorted_null_age_family[["LastName", "Suffix_x", "FirstName_x", "Age_x", "Sex_x",
#                               "Suffix_y", "FirstName_y", "Age_y", "Sex_y"]])

# Now that we have those missing ages that have relatives onboard we can guess the missing ages based on the relatives
# For each last name if there are children and the suffix of the people missing ages are Mr or Mrs, we will replace with
# the median age plus the child's age. If the person missing an age is an adult and their suffix is Mr/Mrs and the other
# person onboard has an opposite suffix we will assume they are married or siblings if female age will be males - XX. If
# same suffix then check if parch or sibsp are non-zero, if parch then assume mother/daugther - if sibsp and suffix is
# miss assume siblings. For mother/daughter if age is greater than median, replace other with their age - median age. If
# sibling then based on if the age is greater or less than median we will subtract or add 2 years.

last_names = sorted_null_age_family["LastName"].unique()
# print(last_names)

for name in last_names:
    print(f"{filtered_null_age[filtered_null_age['LastName'] == name]}\n")









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

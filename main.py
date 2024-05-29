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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
data["Ticket"] = data["Ticket"].str.replace(' ', '')
data["Ticket"] = data["Ticket"].str.replace('/', '')
data["Ticket"] = data["Ticket"].str.replace('.', '')
data[["TicketAlpha", "TicketNumber"]] = data["Ticket"].str.extract(r'([A-Za-z]+)?(\d+)')
data["TicketAlpha"] = data["TicketAlpha"].str.upper()
data["TicketAlpha"] = data["TicketAlpha"].fillna("NONE")
data.to_csv("Data/preprocessed_train.csv")

# cabin_suffix = data["Cabin"].str[0]
# data["CabinSuffix"] = cabin_suffix
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
median_age_by_suffix = data.groupby("Suffix")["Age"].median()
missing_age_mask = data["Age"].isna()
data.loc[missing_age_mask, "Age"] = data.loc[missing_age_mask, "Suffix"].map(median_age_by_suffix)
# class_survival_rate = data.groupby(["Pclass", "TicketAlpha"])["Survived"].mean()
# print(f"Survival rate by class: \n{class_survival_rate}")

# Correlation matrix
# numerical_data = data.select_dtypes(include=["int", "float"])
# correlation_matrix = numerical_data.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Matrix")
# plt.show()

# -------------------------------------- BASELINE MODEL ------------------------------------ #
# # Baseline only uses numerical features
# base_data = data.dropna()
# print(base_data.head(50))
# miss_median_age = base_data.loc[data["Suffix"] == "Miss", "Age"].median(skipna=True)
# print(miss_median_age)
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
features = ["Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Sex",
            "TicketAlpha",
            # "TicketNumber",
            # "LastName",
            # "CabinSuffix",
            # "Embarked",
            # "Ticket",
            # "Cabin",
            "Suffix"
            ]

base_data = data #data.dropna()
X = (base_data[features]) #.values
y = base_data["Survived"] #.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
average_column = data.groupby('TicketAlpha')['Survived'].mean().reset_index()
# print(average_column)
# categories_to_keep = average_column[(average_column["Survived"] < (1/3)) | (average_column["Survived"] > (2/3))]
# categories_to_keep = categories_to_keep["TicketAlpha"].tolist()
# print(categories_to_keep)
low_survival = average_column[(average_column["Survived"] <= (1/3))]
high_survival = average_column[(average_column["Survived"] >= (2/3))]
# print("Low survival")
# print(low_survival["TicketAlpha"].tolist())
# print("High survival")
# print(high_survival["TicketAlpha"].tolist())

encoder = OneHotEncoder(drop='first', handle_unknown="ignore")
encoder.fit(data[["TicketAlpha"]])
# indices_to_keep = data[data["TicketAlpha"].isin(categories_to_keep)].index
encoded_data = encoder.transform(data[["TicketAlpha"]])
# encoded_data_filtered = encoded_data[indices_to_keep]
# print(encoded_data_filtered)

preprocessor = ColumnTransformer(
    transformers=[
        ('ship', OneHotEncoder(handle_unknown="ignore"), ["Sex"]),
        ('sink', encoder, ["TicketAlpha"]),
        # ('sink', OneHotEncoder(handle_unknown="ignore"), ["TicketAlpha"]),
        ('lifeboat', StandardScaler(), ["Fare"]),
        ('ice', StandardScaler(), ["Parch"]),
        ('burg', StandardScaler(), ["SibSp"]),
        ('white', StandardScaler(), ["Age"]),
        ('star', OneHotEncoder(handle_unknown="ignore"), ["Suffix"])
    ],
    remainder='passthrough'  # Keep any remaining columns not specified in transformers as they are
)

# base_data_transformed = preprocessor.fit_transform(base_data)
# print(base_data_transformed[0])
# base_data_processed = base_data_transformed[indices_to_keep]
# print(base_data_processed)

X_train_preprocessed = preprocessor.fit_transform(X_train)
# X_train_preprocessed = X_train_preprocessed.dropna()
# print(pd.DataFrame(X_train_preprocessed))
X_test_preprocessed = preprocessor.transform(X_test)



# -------------------------------------- LOGISTIC REGRESSION FIT ------------------------------------ #
model.fit(X_train_preprocessed, y_train)

lr_y_pred = model.predict(X_test_preprocessed)

lr_accuracy = accuracy_score(y_test, lr_y_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy}")  # Accuracy: 0.6756756756756757
# **************** After removing Embarked, Ticket, and Cabin Accuracy is 0.8108108108108109 **************** #
# coefficients = model.coef_[0]
# for feature, coef in zip(features, coefficients):
#     print(f"Feature {feature}: {coef}")
# print(data["Cabin"].unique())

# -------------------------------------- DECISION TREE FIT ------------------------------------ #
clf = DecisionTreeClassifier()
clf.fit(X_train_preprocessed, y_train)
clf_y_pred = clf.predict(X_test_preprocessed)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
print(f"Decision Tree Accuracy: {clf_accuracy}")

# -------------------------------------- RANDOM FOREST FIT ------------------------------------ #
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train_preprocessed, y_train)
rf_y_pred = rf.predict(X_test_preprocessed)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Accuracy: {rf_accuracy}")

# plt.figure()
# plt.scatter(base_data["Survived"], base_data["TicketAlpha"])

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


test_data = pd.read_csv("Data/test.csv")
median_fare = data["Fare"].median()
test_data["Fare"] = test_data["Fare"].fillna(median_fare)

test_data[["LastName", "SuffixFirstName"]] = test_data["Name"].str.split(",", expand=True)
test_data[["Suffix", "FirstName"]] = test_data["SuffixFirstName"].str.split(".", n=1, expand=True)
test_data.drop(columns=["Name", "SuffixFirstName"], inplace=True)
missing_age_mask = test_data["Age"].isna()
test_data.loc[missing_age_mask, "Age"] = test_data.loc[missing_age_mask, "Suffix"].map(median_age_by_suffix)
test_data["Ticket"] = test_data["Ticket"].str.replace(' ', '')
test_data["Ticket"] = test_data["Ticket"].str.replace('/', '')
test_data["Ticket"] = test_data["Ticket"].str.replace('.', '')
test_data[["TicketAlpha", "TicketNumber"]] = test_data["Ticket"].str.extract(r'([A-Za-z]+)?(\d+)')
test_data["TicketAlpha"] = test_data["TicketAlpha"].str.upper()
test_data["TicketAlpha"] = test_data["TicketAlpha"].fillna("NONE")
# print(test_data.describe())
X_test = test_data[features]
X_test_preprocessed = preprocessor.transform(X_test)

# y_pred = model.predict(X_test_preprocessed)
# y_pred = rf.predict(X_test_preprocessed)
# test_data["Survived"] = y_pred
# predictions = test_data[["PassengerId", "Survived"]]
# print(predictions)
# predictions.to_csv("Data/submission.csv")
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy}")  # Accuracy: 0.6756756756756757



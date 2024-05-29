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
from math import log
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------------------- IMPORT DATA AND PREPROCESS ------------------------------------ #
data = pd.read_csv("Data/train.csv")
# data["Survived"] = data["Survived"].astype(bool)
median_age = data["Age"].median()  # 28
median_fare = data["Fare"].median()
# print(data.describe())


def data_processing(data_frame):
    missing_age_mask = data_frame["Age"].isna()
    data_frame["Fare"] = data_frame["Fare"].fillna(median_fare)
    data_frame["PassengerId"] = data_frame["PassengerId"].astype(object)
    data_frame[["LastName", "SuffixFirstName"]] = data_frame["Name"].str.split(",", expand=True)
    data_frame[["Suffix", "FirstName"]] = data_frame["SuffixFirstName"].str.split(".", n=1, expand=True)
    data_frame.drop(columns=["Name", "SuffixFirstName"], inplace=True)

    data_frame["Ticket"] = data_frame["Ticket"].str.replace(' ', '')
    data_frame["Ticket"] = data_frame["Ticket"].str.replace('/', '')
    data_frame["Ticket"] = data_frame["Ticket"].str.replace('.', '')
    data_frame[["TicketAlpha", "TicketNumber"]] = data_frame["Ticket"].str.extract(r'([A-Za-z]+)?(\d+)')
    data_frame["TicketAlpha"] = data_frame["TicketAlpha"].str.upper()
    data_frame["TicketAlpha"] = data_frame["TicketAlpha"].fillna("NONE")

    # data["FamilyScore"] =


data_processing(data)
median_age_by_suffix = data.groupby("Suffix")["Age"].median()


def impute_age(data_frame):
    missing_age_mask = data_frame["Age"].isna()
    data_frame.loc[missing_age_mask, "Age"] = data_frame.loc[missing_age_mask, "Suffix"].map(median_age_by_suffix)


impute_age(data)
# print(data.dtypes)


def family_score(data_frame):
    data_frame["FamilyScore"] = ((((data_frame["SibSp"] + 1) / (data_frame["Parch"] + 1)) / data_frame["Age"])
                                 * data_frame["Pclass"])
    # print(data_frame.sort_values("FamilyScore"))


family_score(data)
# data.to_csv("Data/processed_train.csv")

# -------------------------------------- BASIC ANALYSIS ------------------------------------ #
# From simple correlation analysis it looks like fare and class influenced survival the most. The higher the fare cost
# the less likely you were to die, the lower the class (higher the numerical value) the lower the chance of survival. Of
# course these two variables are also highly correlated. As are parch and sibsp

# # Correlation matrix
# numerical_data = data.select_dtypes(include=["int", "float"])
# correlation_matrix = numerical_data.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Matrix")
# plt.show()

# -------------------------------------- PREPROCESSING ------------------------------------ #


features = ["Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "FamilyScore",
            "Sex",
            "TicketAlpha",
            "Suffix"
            ]

base_data = data #data.dropna()
X = (base_data[features]) #.values
y = base_data["Survived"] #.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)
average_column = data.groupby('TicketAlpha')['Survived'].mean().reset_index()

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
        ('nomen', StandardScaler(), ["FamilyScore"]),
        ('lifeboat', StandardScaler(), ["Fare"]),
        ('ice', StandardScaler(), ["Parch"]),
        ('burg', StandardScaler(), ["SibSp"]),
        ('white', StandardScaler(), ["Age"]),
        ('star', OneHotEncoder(handle_unknown="ignore"), ["Suffix"])
    ],
    remainder='passthrough'  # Keep any remaining columns not specified in transformers as they are
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# -------------------------------------- LOGISTIC REGRESSION FIT ------------------------------------ #
model = LogisticRegression()
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

plt.figure()
mean_score = base_data["FamilyScore"].mean()
plt.scatter(base_data["FamilyScore"], base_data["Survived"])
plt.show()


test_data = pd.read_csv("Data/test.csv")
data_processing(test_data)
impute_age(test_data)
family_score(test_data)

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



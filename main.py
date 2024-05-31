"""
Project: Kaggle - Titanic Survival Prediction
Start: May 28th, 2024
Last touched: May 31st, 2024
Author: Madeleine L.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import log
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# -------------------------------------- IMPORT DATA AND PREPROCESS ------------------------------------ #
data = pd.read_csv("Data/train.csv")
test_data = pd.read_csv("Data/test.csv")
# median_age = data["Age"].median()  # 28
# median_fare = data["Fare"].median()
# print(data.describe())


def data_processing(data_frame):
    """Takes dataframe as input and performs basic data processing and feature engineering."""

    data_frame["PassengerId"] = data_frame["PassengerId"].astype(object)
    data_frame[["LastName", "SuffixFirstName"]] = data_frame["Name"].str.split(",", expand=True)
    data_frame[["Suffix", "FirstName"]] = data_frame["SuffixFirstName"].str.split(".", n=1, expand=True)
    data_frame.drop(columns=["Name", "SuffixFirstName"], inplace=True)

    # Simplify and clean Ticket feature
    data_frame["Ticket"] = data_frame["Ticket"].str.replace(' ', '')
    data_frame["Ticket"] = data_frame["Ticket"].str.replace('/', '')
    data_frame["Ticket"] = data_frame["Ticket"].str.replace('.', '')
    data_frame[["TicketAlpha", "TicketNumber"]] = data_frame["Ticket"].str.extract(r'([A-Za-z]+)?(\d+)')
    data_frame["TicketAlpha"] = data_frame["TicketAlpha"].str.upper()
    data_frame["TicketAlpha"] = data_frame["TicketAlpha"].fillna("Missing")

    # Simplify and clean Cabin feature
    data_frame["CabinAlpha"] = data["Cabin"].str.extract(r'([A-Za-z])')
    data_frame["CabinAlpha"] = data_frame["CabinAlpha"].fillna("Missing")
    data_frame["CabinAlpha"] = data_frame["CabinAlpha"].replace(["A", "B", "C", "T"], "ABC")
    data_frame["CabinAlpha"] = data_frame["CabinAlpha"].replace(["D", "E"], "DE")
    # data_frame["CabinAlpha"] = data_frame["CabinAlpha"].replace(["F", "G"], "FG")



data_processing(data)
data_processing(test_data)

median_age_by_suffix = data.groupby("Suffix")["Age"].median()
median_age_by_pclass = data.groupby("Pclass")["Age"].median()
median_fare_by_pclass = data.groupby("Pclass")["Fare"].median()


def impute_age_fare(data_frame):
    """Takes a dataframe as input and imputes the age of null records based on median age calculated by pclass"""
    missing_age_mask = data_frame["Age"].isna()
    # data_frame.loc[missing_age_mask, "Age"] = data_frame.loc[missing_age_mask, "Suffix"].map(median_age_by_suffix)
    data_frame.loc[missing_age_mask, "Age"] = data_frame.loc[missing_age_mask, "Pclass"].map(median_age_by_pclass)

    missing_fare_mask = data_frame["Fare"].isna()
    data_frame.loc[missing_fare_mask, "Fare"] = data_frame.loc[missing_fare_mask, "Pclass"].map(median_fare_by_pclass)


impute_age_fare(data)
impute_age_fare(test_data)


def family_score(data_frame):
    """Takes a dataframe as input and creates a family score to capture the complexities of having family onboard"""
    data_frame["FamilyScore"] = ((((data_frame["SibSp"] + 1) / (data_frame["Parch"] + 1)) * data_frame["Age"])
                                 / (data_frame["Fare"] + 1))
    # print(data_frame.sort_values("FamilyScore"))


family_score(data)
family_score(test_data)

# -------------------------------------- DATA EXPLORATION ------------------------------------ #
# From simple correlation analysis it looks like fare and class influenced survival the most. The higher the fare cost
# the less likely you were to die, the lower the class (higher the numerical value) the lower the chance of survival. Of
# course these two variables are also highly correlated. As are parch and sibsp

# print(data.info())
# print(test_data.info())

# # Encode Suffix to gauge correlation with age
# encoder = LabelEncoder()
# encoder.fit(data[["Suffix"]])
# encoded_data = encoder.transform(data[["Suffix"]])
# data["EncodedSuffix"] = encoded_data
# print(data.info())

# # Encode CabinAlpha to gauge correlation with survived
# encoder = LabelEncoder()
# encoder.fit(data[["CabinAlpha"]])
# encoded_data = encoder.transform(data[["CabinAlpha"]])
# data["EncodedCabin"] = encoded_data
# # print(data.info())
#
# # Correlation matrix
# numerical_data = data.select_dtypes(include=["int", "float"])
# correlation_matrix = numerical_data.corr()
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title("Correlation Matrix")
# plt.show()

# grouped = data.groupby(['TicketAlpha']).size().reset_index(name='count')
# filtered_groups = grouped[grouped['count'] > 2]
# merged_data = pd.merge(filtered_groups, data, on=['TicketAlpha'], how='inner')
# mean_survival_rate = merged_data.groupby(['TicketAlpha'])['Survived'].mean().reset_index(name='MeanSurvivalRate')
# low_survival = mean_survival_rate[mean_survival_rate["MeanSurvivalRate"] <= 0.40]
# low_survival_list = low_survival["TicketAlpha"].tolist()
#
#
# # def low_survival_tickets(row):
# #     if row["TicketAlpha"] in low_survival_list:
# #         return 1
# #     else:
# #         return 0
# #
# #
# # data["LowSurvival"] = data.apply(low_survival_tickets, axis=1)
#
# # print(data.head(50))

cabin_survival = data["Survived"].groupby(data["CabinAlpha"]).mean()
print(cabin_survival)
# cabin_count = data["CabinAlpha"].groupby(data["CabinAlpha"]).count()
# print(cabin_count)
# cabin_sum = data["Survived"].groupby(data["CabinAlpha"]).sum()
# print(cabin_sum)

# grouped_data = data.groupby(['CabinAlpha', 'Pclass']).sum().reset_index()
# sns.barplot(x="CabinAlpha", y="Survived", hue="Pclass", data=grouped_data)
# plt.title("Survived by cabin and class")
# plt.xlabel("CabinAlpha")
# plt.ylabel("Survived")
# plt.show()

grouped_data = data.groupby(['TicketAlpha', 'Pclass']).sum().reset_index()
sns.barplot(x="Pclass", y="TicketAlpha", hue="Survived", data=grouped_data)
plt.title("Survived by cabin and class")
plt.xlabel("Pclass")
plt.ylabel("TicketAlpha")
plt.show()
# -------------------------------------- PREPROCESSING ------------------------------------ #


features = ["Pclass",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "FamilyScore",
            "Sex",
            # "TicketAlpha",
            "CabinAlpha"
            ]

base_data = data #data.dropna()
X = (base_data[features]) #.values
y = base_data["Survived"] #.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

# average_column = data.groupby('TicketAlpha')['Survived'].mean().reset_index()

# low_survival = average_column[(average_column["Survived"] <= (1/3))]
# high_survival = average_column[(average_column["Survived"] >= (2/3))]
# print("Low survival")
# print(low_survival["TicketAlpha"].tolist())
# print("High survival")
# print(high_survival["TicketAlpha"].tolist())

# encoder = OneHotEncoder(drop='first', handle_unknown="ignore")
# encoder.fit(data[["TicketAlpha"]])
# # indices_to_keep = data[data["TicketAlpha"].isin(categories_to_keep)].index
# encoded_data = encoder.transform(data[["TicketAlpha"]])
# # encoded_data_filtered = encoded_data[indices_to_keep]
# # print(encoded_data_filtered)

preprocessor = ColumnTransformer(
    transformers=[
        ('ship', OneHotEncoder(handle_unknown="ignore"), ["Sex"]),
        ('level', OneHotEncoder(), ['CabinAlpha']),
        # ('sink', OneHotEncoder(handle_unknown="ignore"), ["TicketAlpha"]),
        ('nomen', StandardScaler(), ["FamilyScore"]),
        ('lifeboat', StandardScaler(), ["Fare"]),
        ('ice', StandardScaler(), ["Parch"]),
        ('burg', StandardScaler(), ["SibSp"]),
        ('white', StandardScaler(), ["Age"])
        # ('star', StandardScaler(), ["LowSurvival"])
    ]
    , remainder='passthrough'  # Keep any remaining columns not specified in transformers as they are
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)


# pca = PCA()
# X_pca = pca.fit_transform(X_train_preprocessed)
#
# # Fit PCA to the preprocessed training data
# pca.fit(X_train_preprocessed)
#
# # Transform the preprocessed training and test data using the fitted PCA
# X_train_pca = pca.transform(X_train_preprocessed)
# X_test_pca = pca.transform(X_test_preprocessed)
#
# explained_variance_ratio = pca.explained_variance_ratio_
# print("Explained Variance Ratio:", explained_variance_ratio)
#
# # -------------------------------------- LOGISTIC REGRESSION FIT ------------------------------------ #
model = LogisticRegression()
model.fit(X_train_preprocessed, y_train)

lr_y_pred = model.predict(X_test_preprocessed)

lr_accuracy = accuracy_score(y_test, lr_y_pred)
print(f"Logistic Regression Accuracy: {lr_accuracy}")  # Accuracy: 0.6756756756756757
# # **************** After removing Embarked, Ticket, and Cabin Accuracy is 0.8108108108108109 **************** #
# # coefficients = model.coef_[0]
# # for feature, coef in zip(features, coefficients):
# #     print(f"Feature {feature}: {coef}")
# # print(data["Cabin"].unique())
#
# # -------------------------------------- DECISION TREE FIT ------------------------------------ #
clf = DecisionTreeClassifier()
clf.fit(X_train_preprocessed, y_train)
clf_y_pred = clf.predict(X_test_preprocessed)
clf_accuracy = accuracy_score(y_test, clf_y_pred)
print(f"Decision Tree Accuracy: {clf_accuracy}")
#
# # -------------------------------------- RANDOM FOREST FIT ------------------------------------ #
rf = RandomForestClassifier(n_estimators=1000)
rf.fit(X_train_preprocessed, y_train)
rf_y_pred = rf.predict(X_test_preprocessed)
rf_accuracy = accuracy_score(y_test, rf_y_pred)
print(f"Random Forest Accuracy: {rf_accuracy}")
#
# # plt.figure()
# # mean_score = base_data["FamilyScore"].mean()
# # plt.scatter(base_data["FamilyScore"], base_data["Survived"])
# # plt.show()
#
# # test_data["LowSurvival"] = test_data.apply(low_survival_tickets, axis=1)
#
# X_test = test_data[features]
# X_test_preprocessed = preprocessor.transform(X_test)
#
# # y_pred = model.predict(X_test_preprocessed)
# # y_pred = rf.predict(X_test_preprocessed)
# # test_data["Survived"] = y_pred
# # predictions = test_data[["PassengerId", "Survived"]]
# # print(predictions)
# # predictions.to_csv("Data/submission.csv")
# # accuracy = accuracy_score(y_test, y_pred)
# # print(f"Accuracy: {accuracy}")  # Accuracy: 0.6756756756756757



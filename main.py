import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Function removes all the punctuation, special characters and capital letters
def text_cleaning(row):
    text = re.sub('[^a-zA-Z]', ' ', row).lower()
    clean_text = " ".join(text.split())

    return clean_text


# Reading the data file
df = pd.read_csv('tweet_data.csv', encoding='latin-1')
print(f"There are {len(df)} datapoints before filtering.")

# Filtering the dataframe:
#   1. Remove all columns but 2
#   2. Drop all the rows with Nan values
#   3. Renaming columns
df = df.filter(["user_timezone", "text"])
df.dropna(axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.rename(columns={"user_timezone": "timezone"}, inplace=True)
print(f"There are {len(df)} clean datapoints before filtering.")

# Conversion into UTC time zones
df.loc[df["timezone"] == "Cape Verde Is.", "timezone"] = "UTC-1"
df.loc[df["timezone"] == "Newfoundland", "timezone"] = "UTC-3"
df.loc[df["timezone"] == "Atlantic Time (Canada)", "timezone"] = "UTC-3"
df.loc[df["timezone"] == "Brasilia", "timezone"] = "UTC-3"
df.loc[df["timezone"] == "Buenos Aires", "timezone"] = "UTC-3"
df.loc[df["timezone"] == "America/Argentina/Buenos_Aires", "timezone"] = "UTC-3"
df.loc[df["timezone"] == "Georgetown", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "Caracas", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "Indiana (East)", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "EDT", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "Santiago", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "Mid-Atlantic", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "America/Toronto", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "America/New_York", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "America/Detroit", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "La Paz", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "Eastern Time (US & Canada)", "timezone"] = "UTC-4"
df.loc[df["timezone"] == "Guadalajara", "timezone"] = "UTC-5"
df.loc[df["timezone"] == "America/Chicago", "timezone"] = "UTC-5"
df.loc[df["timezone"] == "CDT", "timezone"] = "UTC-5"
df.loc[df["timezone"] == "Mexico City", "timezone"] = "UTC-5"
df.loc[df["timezone"] == "Lima", "timezone"] = "UTC-5"
df.loc[df["timezone"] == "Quito", "timezone"] = "UTC-5"
df.loc[df["timezone"] == "Central Time (US & Canada)", "timezone"] = "UTC-5"
df.loc[df["timezone"] == "Bogota", "timezone"] = "UTC-5"
df.loc[df["timezone"] == "Monterrey", "timezone"] = "UTC-5"
df.loc[df["timezone"] == "America/Boise", "timezone"] = "UTC-6"
df.loc[df["timezone"] == "Arizona", "timezone"] = "UTC-6"
df.loc[df["timezone"] == "CST", "timezone"] = "UTC-6"
df.loc[df["timezone"] == "Chihuahua", "timezone"] = "UTC-6"
df.loc[df["timezone"] == "Mountain Time (US & Canada)", "timezone"] = "UTC-6"
df.loc[df["timezone"] == "Central America", "timezone"] = "UTC-6"
df.loc[df["timezone"] == "Saskatchewan", "timezone"] = "UTC-6"
df.loc[df["timezone"] == "Mazatlan", "timezone"] = "UTC-6"
df.loc[df["timezone"] == "America/Vancouver", "timezone"] = "UTC-7"
df.loc[df["timezone"] == "Pacific Time (US & Canada)", "timezone"] = "UTC-7"
df.loc[df["timezone"] == "America/Los_Angeles", "timezone"] = "UTC-7"
df.loc[df["timezone"] == "PDT", "timezone"] = "UTC-7"
df.loc[df["timezone"] == "Tijuana", "timezone"] = "UTC-7"
df.loc[df["timezone"] == "PST", "timezone"] = "UTC-8"
df.loc[df["timezone"] == "Alaska", "timezone"] = "UTC-8"
df.loc[df["timezone"] == "Hawaii", "timezone"] = "UTC-10"
df.loc[df["timezone"] == "Midway Island", "timezone"] = "UTC-11"
df.loc[df["timezone"] == "International Date Line West", "timezone"] = "UTC-15"
df.loc[df["timezone"] == "Azores", "timezone"] = "UTC"
df.loc[df["timezone"] == "Greenland", "timezone"] = "UTC"
df.loc[df["timezone"] == "Monrovia", "timezone"] = "UTC"
df.loc[df["timezone"] == "Africa/Lagos", "timezone"] = "UTC+1"
df.loc[df["timezone"] == "BST", "timezone"] = "UTC+1"
df.loc[df["timezone"] == "West Central Africa", "timezone"] = "UTC+1"
df.loc[df["timezone"] == "Lisbon", "timezone"] = "UTC+1"
df.loc[df["timezone"] == "Casablanca", "timezone"] = "UTC+1"
df.loc[df["timezone"] == "Edinburgh", "timezone"] = "UTC+1"
df.loc[df["timezone"] == "London", "timezone"] = "UTC+1"
df.loc[df["timezone"] == "Europe/London", "timezone"] = "UTC+1"
df.loc[df["timezone"] == "Dublin", "timezone"] = "UTC+1"
df.loc[df["timezone"] == "Europe/Sarajevo", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Africa/Cairo", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Ljubljana", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Harare", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Bern", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Zagreb", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Cairo", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Budapest", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Stockholm", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Bratislava", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Skopje", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Copenhagen", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Prague", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Madrid", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Amsterdam", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Rome", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Pretoria", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Paris", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Europe/Paris", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Belgrade", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Warsaw", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Berlin", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Paris", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Brussels", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Vienna", "timezone"] = "UTC+2"
df.loc[df["timezone"] == "Europe/Moscow"] = "UTC+3"
df.loc[df["timezone"] == "Baghdad"] = "UTC+3"
df.loc[df["timezone"] == "Moscow"] = "UTC+3"
df.loc[df["timezone"] == "Athens"] = "UTC+3"
df.loc[df["timezone"] == "Europe/Athens"] = "UTC+3"
df.loc[df["timezone"] == "Jerusalem"] = "UTC+3"
df.loc[df["timezone"] == "Bucharest"] = "UTC+3"
df.loc[df["timezone"] == "Istanbul"] = "UTC+3"
df.loc[df["timezone"] == "Helsinki"] = "UTC+3"
df.loc[df["timezone"] == "Nairobi"] = "UTC+3"
df.loc[df["timezone"] == "Africa/Nairobi"] = "UTC+3"
df.loc[df["timezone"] == "Volgograd"] = "UTC+3"
df.loc[df["timezone"] == "Kyiv"] = "UTC+3"
df.loc[df["timezone"] == "Vilnius"] = "UTC+3"
df.loc[df["timezone"] == "GMT+3"] = "UTC+3"
df.loc[df["timezone"] == "Minsk"] = "UTC+3"
df.loc[df["timezone"] == "Tallinn"] = "UTC+3"
df.loc[df["timezone"] == "Riyadh"] = "UTC+3"
df.loc[df["timezone"] == "Riga"] = "UTC+3"
df.loc[df["timezone"] == "Sofia"] = "UTC+3"
df.loc[df["timezone"] == "Kuwait"] = "UTC+3"
df.loc[df["timezone"] == "Baku"] = "UTC+4"
df.loc[df["timezone"] == "Abu Dhabi"] = "UTC+4"
df.loc[df["timezone"] == "Yerevan"] = "UTC+4"
df.loc[df["timezone"] == "Muscat"] = "UTC+4"
df.loc[df["timezone"] == "Kabul"] = "UTC+4:30"
df.loc[df["timezone"] == "Tehran"] = "UTC+4:30"
df.loc[df["timezone"] == "Ekaterinburg"] = "UTC+5"
df.loc[df["timezone"] == "Tashkent"] = "UTC+5"
df.loc[df["timezone"] == "Islamabad"] = "UTC+5"
df.loc[df["timezone"] == "Asia/Karachi"] = "UTC+5"
df.loc[df["timezone"] == "Karachi"] = "UTC+5"
df.loc[df["timezone"] == "IST"] = "UTC+5:30"
df.loc[df["timezone"] == "Mumbai"] = "UTC+5:30"
df.loc[df["timezone"] == "New Delhi"] = "UTC+5:30"
df.loc[df["timezone"] == "Chennai"] = "UTC+5:30"
df.loc[df["timezone"] == "Sri Jayawardenepura"] = "UTC+5:30"
df.loc[df["timezone"] == "Kolkata"] = "UTC+5:30"
df.loc[df["timezone"] == "Dhaka"] = "UTC+6"
df.loc[df["timezone"] == "Almaty"] = "UTC+6"
df.loc[df["timezone"] == "Novosibirsk"] = "UTC+7"
df.loc[df["timezone"] == "Hanoi"] = "UTC+7"
df.loc[df["timezone"] == "Jakarta"] = "UTC+7"
df.loc[df["timezone"] == "Bangkok"] = "UTC+7"
df.loc[df["timezone"] == "Krasnoyarsk"] = "UTC+7"
df.loc[df["timezone"] == "Ulaan Bataar"] = "UTC+8"
df.loc[df["timezone"] == "Urumqi"] = "UTC+8"
df.loc[df["timezone"] == "Kuala Lumpur"] = "UTC+8"
df.loc[df["timezone"] == "Perth"] = "UTC+8"
df.loc[df["timezone"] == "Irkutsk"] = "UTC+8"
df.loc[df["timezone"] == "Singapore"] = "UTC+8"
df.loc[df["timezone"] == "Hong Kong"] = "UTC+8"
df.loc[df["timezone"] == "Taipei"] = "UTC+8"
df.loc[df["timezone"] == "Beijing"] = "UTC+8"
df.loc[df["timezone"] == "Osaka"] = "UTC+9"
df.loc[df["timezone"] == "Tokyo"] = "UTC+9"
df.loc[df["timezone"] == "Seoul"] = "UTC+9"
df.loc[df["timezone"] == "Yakutsk"] = "UTC+9"
df.loc[df["timezone"] == "Darwin"] = "UTC+9:30"
df.loc[df["timezone"] == "Adelaide"] = "UTC+9:30"
df.loc[df["timezone"] == "Australia/Brisbane"] = "UTC+10"
df.loc[df["timezone"] == "Canberra"] = "UTC+10"
df.loc[df["timezone"] == "Sydney"] = "UTC+10"
df.loc[df["timezone"] == "Hobart"] = "UTC+10"
df.loc[df["timezone"] == "Melbourne"] = "UTC+10"
df.loc[df["timezone"] == "Brisbane"] = "UTC+10"
df.loc[df["timezone"] == "Guam"] = "UTC+10"
df.loc[df["timezone"] == "Magadan"] = "UTC+11"
df.loc[df["timezone"] == "Solomon Is."] = "UTC+11"
df.loc[df["timezone"] == "New Caledonia"] = "UTC+11"
df.loc[df["timezone"] == "Fiji"] = "UTC+12"
df.loc[df["timezone"] == "Wellington"] = "UTC+12"
df.loc[df["timezone"] == "Auckland"] = "UTC+12"
df.loc[df["timezone"] == "Samoa"] = "UTC+13"
df.loc[df["timezone"] == "Nuku'alofa"] = "UTC+13"

# Counting datapoints for each time zone
# temp = df.groupby(["timezone"]).count()
# temp = temp.sort_values(by=["text"], ascending=True)
# for i in range(len(np.unique(df["timezone"]))):
#     print(i, temp.iloc[i])
# print()

# Keep all labels with more than 100 datapoints
timezones_to_keep = ["UTC-7", "UTC-4", "UTC-5", "UTC+1", "UTC+2", "UTC-3", "UTC-6",
                     "UTC+3", "UTC-10", "UTC+8", "UTC-8", "UTC+10"]
df.drop(df[~df["timezone"].isin(timezones_to_keep)].index, inplace=True)
print(f"There are {len(df)} datapoints after filtering.")

# Encoding timezones with integer values starting from 0
le = LabelEncoder()
df["timezone"] = le.fit_transform(df["timezone"])

# Cleaning the text from uppercase letters, special characters and punctuation
df["clean_text"] = df["text"].apply(lambda f: text_cleaning(f))
df.drop(["text"], axis=1, inplace=True)

# Take n features:
#   1. Creating CountVectorizer()
#   2. Counting the number of each word in the whole dataset
#   3. Get the vocabulary and put into the "names" variable
#   4. Creating the dataframe from vocabulary and the number of words in the dataset
#   5. Sorting the dataframe by the number of words in the dataset
#   6. Choosing n_features most frequent words and adding the rest of them to the stop_words list
count_vectorizer = CountVectorizer()
counter = count_vectorizer.fit_transform(df["clean_text"]).toarray()
word_frequency = counter.sum(axis=0)
names = count_vectorizer.get_feature_names_out()
data = pd.DataFrame({"name": names,
                     "frequency": word_frequency})
data = data.sort_values(by=["frequency"])
n_features = 2000
stop_words = data.head(len(names) - n_features)["name"].to_numpy()
print(f"There are {len(names)} features (unique words) in total, but we get only {n_features} out of them")

# Convert the text features to the TF-IDF features
# X_train is a rxc matrix, where r is the number of data points and c is the number of unique words
tf_idf_vectorizer = TfidfVectorizer(stop_words=list(stop_words))
X = tf_idf_vectorizer.fit_transform(df["clean_text"]).toarray()
print(f"The size of each feature matrix is {len(X)} x {len(X[0])}")

# Get the numpy array of labels
y = df["timezone"].to_numpy()
print(f"There are {len(np.unique(y))} classes (timezones)\n")

# Random training/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=3)

# Logistic Regression
log_reg = LogisticRegression(multi_class='ovr')
log_reg.fit(X_train, y_train)

err_train_log_reg = log_reg.score(X_train, y_train)
err_val_log_reg = log_reg.score(X_val, y_val)
print("Logistic regression:")
print(f"Training error: {round(100 * err_train_log_reg, 2)}%")
print(f"Validation error: {round(100 * err_val_log_reg, 2)}%")
print()

# Support Vector Classification with radial-based function as kernel
svm_rbf = SVC(kernel="rbf")
svm_rbf.fit(X_train, y_train)

err_train_svm_rbf = svm_rbf.score(X_train, y_train)
err_val_svm_rbf = svm_rbf.score(X_val, y_val)
print("SVC with rbf kernel:")
print(f"Training error: {round(100 * err_train_svm_rbf, 2)}%")
print(f"Validation error: {round(100 * err_val_svm_rbf, 2)}%")
print()

# Support Vector Classification with polynomial functions as kernel
svm_poly_2 = SVC(kernel="poly", degree=2)
svm_poly_2.fit(X_train, y_train)

err_train_svm_poly_2 = svm_poly_2.score(X_train, y_train)
err_val_svm_poly_2 = svm_poly_2.score(X_val, y_val)
print("SVC with 2nd degree polynomial kernel:")
print(f"Training error: {round(100 * err_train_svm_poly_2, 2)}%")
print(f"Validation error: {round(100 * err_val_svm_poly_2, 2)}%")
print()

svm_poly_3 = SVC(kernel="poly", degree=3)
svm_poly_3.fit(X_train, y_train)

err_train_svm_poly_3 = svm_poly_3.score(X_train, y_train)
err_val_svm_poly_3 = svm_poly_3.score(X_val, y_val)
print("SVC with 3rd degree polynomial kernel:")
print(f"Training error: {round(100 * err_train_svm_poly_3, 2)}%")
print(f"Validation error: {round(100 * err_val_svm_poly_3, 2)}%")
print()

svm_poly_4 = SVC(kernel="poly", degree=4)
svm_poly_4.fit(X_train, y_train)

err_train_svm_poly_4 = svm_poly_4.score(X_train, y_train)
err_val_svm_poly_4 = svm_poly_4.score(X_val, y_val)
print("SVC with 4th degree polynomial kernel:")
print(f"Training error: {round(100 * err_train_svm_poly_4, 2)}%")
print(f"Validation error: {round(100 * err_val_svm_poly_4, 2)}%")
print()

# Plotting training and validation scores of each model
training_errors = [err_train_log_reg, err_train_svm_rbf, err_train_svm_poly_2, err_train_svm_poly_3,
                   err_train_svm_poly_4]
validation_errors = [err_val_log_reg, err_val_svm_rbf, err_val_svm_poly_2, err_val_svm_poly_3,
                     err_val_svm_poly_4]
regressions = ["Logistic", "SVM_rbf", "SVM_poly_2", "SVM_poly_3", "SVM_poly_4"]

plt.bar(regressions, training_errors, alpha=0.5, color="b", label="Training score")
plt.bar(regressions, validation_errors, alpha=0.5, color="g", label="Validation score")
plt.xlabel("Regression")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Tests obtained in experiments

# Test #1
# The size of feature matrix is 12252 x 200
# Training errors for 5 CV folds are: (array([0.25159158, 0.25134672, 0.25159158]), 2)
# Average training error for 5 CV folds is: 25.15%
# Validation errors for 5 CV folds are: [0.19147894 0.20666014 0.19539667]
# Average validation error for 5 CV folds is: 19.78%
# Training errors for 5 CV folds are: [0.39700031 0.39812264 0.39624566 0.39726586 0.39828606]
# Average training error for 5 CV folds is: 39.74%
# Validation errors for 5 CV folds are: [0.21419829 0.2129743  0.22612245 0.22122449 0.21632653]
# Average validation error for 5 CV folds is: 21.82%

# Test #2
# The size of feature matrix is 12252 x 500
# Logistic regression:
# Training error: 27.98%
# Validation error: 21.66%
#
# SVC:
# Training error: 46.82%
# Validation error: 22.97%

# Test #3
# The size of feature matrix is 10762 x 2000
# Logistic regression:
# Training error: 41.81%
# Validation error: 26.24%
#
# SVC:
# Training error: 63.72%
# Validation error: 28.1%

# Test #4
# There are 20441 features in total, but we get only 2000 out of them
# The size of feature matrix is 6841 x 2000
# There are 3 classes (timezones)
#
# Logistic regression:
# Training error: 63.98%
# Validation error: 39.37%
#
# SVC with rbf kernel:
# Training error: 91.3%
# Validation error: 40.83%
#
# SVC with 3rd degree polynomial kernel:
# Training error: 98.01%
# Validation error: 42.0%

# Test #5
# There are 29156 features in total, but we get only 2000 out of them
# The size of feature matrix is 11775 x 2000
# There are 12 classes (timezones)
#
# Logistic regression:
# Training error: 47.09%
# Validation error: 27.35%
#
# SVC with rbf kernel:
# Training error: 73.16%
# Validation error: 27.86%
#
# SVC with 3rd degree polynomial kernel:
# Training error: 94.63%
# Validation error: 28.2%

from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd

total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv")

total_data.head()

def apply_preprocess(df):
    df = df.drop("package_name", axis=1)
    df["review"] = df["review"].str.strip().str.lower()

    return df

total_data = apply_preprocess(total_data)

total_data.head()

def apply_preprocess(df):
    df = df.drop("package_name", axis=1)
    df["review"] = df["review"].str.strip().str.lower()

    return df

total_data = apply_preprocess(total_data)

total_data.head()

from sklearn.model_selection import train_test_split

X = total_data["review"]
y = total_data["polarity"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train.head()

from sklearn.feature_extraction.text import CountVectorizer

vec_model = CountVectorizer(stop_words = "english")
X_train = vec_model.fit_transform(X_train).toarray()
X_test = vec_model.transform(X_test).toarray()

X_train

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)

from sklearn.naive_bayes import GaussianNB, BernoulliNB

for model_aux in [GaussianNB(), BernoulliNB()]:
    model_aux.fit(X_train, y_train)
    y_pred_aux = model_aux.predict(X_test)
    print(f"{model_aux} with accuracy: {accuracy_score(y_test, y_pred_aux)}")
    

import numpy as np
from sklearn.model_selection import RandomizedSearchCV

hyperparams = {
    "alpha": np.linspace(0.01, 10.0, 200),
    "fit_prior": [True, False]
}

# We initialize the random search
random_search = RandomizedSearchCV(model, hyperparams, n_iter = 50, scoring = "accuracy", cv = 5, random_state = 42)
random_search

random_search.fit(X_train, y_train)

print(f"Best hyperparameters: {random_search.best_params_}")

model = MultinomialNB(alpha = 1.917638190954774, fit_prior = False)
model.fit(X_train, y_train)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)


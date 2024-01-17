import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

from sklearn.utils import class_weight


# Support Vector Machine (SVM) to classify the data
def train_svm_balanced(train, test, save_dir):
    # Handling NaN values in the "text" column
    train["text"].fillna("", inplace=True)
    test["text"].fillna("", inplace=True)

    # Calculate class weights
    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train["class"]), y=train["class"])

    # Update LinearSVC with class weights
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC(class_weight='balanced')),
                         ])

    # Training model
    text_clf.fit(train["text"], train["class"])

    # Predicting test set
    predictions = text_clf.predict(test["text"])

    # Printing results
    print("Accuracy score: ", accuracy_score(test["class"], predictions))
    print("Confusion matrix: \n", confusion_matrix(test["class"], predictions))
    print("Classification report: \n", classification_report(test["class"], predictions))

    # Save the trained model
    with open(save_dir, "wb") as f:
        pickle.dump(text_clf, f)

    return text_clf

def train_svm(train, test, save_dir):
    # Handling NaN values in the "text" column
    train["text"].fillna("", inplace=True)
    test["text"].fillna("", inplace=True)

    # Update LinearSVC with class weights
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC),
                         ])

    # Training model
    text_clf.fit(train["text"], train["class"])

    # Predicting test set
    predictions = text_clf.predict(test["text"])

    # Printing results
    print("Accuracy score: ", accuracy_score(test["class"], predictions))
    print("Confusion matrix: \n", confusion_matrix(test["class"], predictions))
    print("Classification report: \n", classification_report(test["class"], predictions))

    # Save the trained model
    with open(save_dir, "wb") as f:
        pickle.dump(text_clf, f)

    return text_clf


def check_svm(input_text):
    with open("../models/svm_model.pkl", "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([input_text])

    if prediction == 0:
        print("Non-harmful")
    elif prediction == 1:
        print("Harmful")

    return prediction

def check_svm_balanced(input_text):
    with open("../models/svm_model_balanced.pkl", "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([input_text])

    if prediction == 0:
        print("Non-harmful")
    elif prediction == 1:
        print("Harmful")

    return prediction







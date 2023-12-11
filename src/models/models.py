import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Loading data
train = pd.read_csv("../data/Train/train_preprocessed.csv", sep="\t")
test = pd.read_csv("../data/Test/test_preprocessed.csv", sep="\t")

# Support Vector Machine (SVM) to classify the data
def train_svm(train, test, save_dir):
    # Creating pipeline
    text_clf = Pipeline([('tfidf', TfidfVectorizer()),
                         ('clf', LinearSVC()),
                         ])
    # Training model
    text_clf.fit(train["text"], train["class"])
    # Predicting test set
    predictions = text_clf.predict(test["text"])
    # Printing results
    print("Accuracy score: ", accuracy_score(test["class"], predictions))
    print("Confusion matrix: \n", confusion_matrix(test["class"], predictions))
    print("Classification report: \n", classification_report(test["class"], predictions))

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





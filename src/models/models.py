import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

from sklearn.utils import class_weight

from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer


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

def train_flair_classifier(train_file, test_file, save_dir):
    # Define your corpus
    corpus: Corpus = ColumnCorpus(
        data_folder='.',  # The directory containing the train and test files
        column_format={0: 'text', 1: 'class'},
        train_file=train_file,
        test_file=test_file
    )

    # Define embeddings (you can use any combination of embeddings)
    word_embeddings = WordEmbeddings('glove')
    flair_forward_embeddings = FlairEmbeddings('polish-forward')

    # Combine the embeddings
    document_embeddings = DocumentRNNEmbeddings([word_embeddings, flair_forward_embeddings])

    # Create the label dictionary from the corpus
    label_dictionary = corpus.make_label_dictionary('class')

    # Define the text classifier model
    classifier = TextClassifier(
        embeddings=document_embeddings,
        label_dictionary=label_dictionary,
        label_type='class'
    )

    # Define the trainer
    trainer = ModelTrainer(classifier, corpus)

    # Train the model
    trainer.train(save_dir, max_epochs=3)

    # Save the model
    classifier.save(save_dir)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Multinomial Naive Bayes model
def train_mnb(train, test, save_dir):
    train["text"].fillna("", inplace=True)
    test["text"].fillna("", inplace=True)

    # Data vectorizing
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])

    # Training model
    model = MultinomialNB()
    model.fit(X_train, train["class"])

    # Predicting test set
    predictions = model.predict(X_test)

    # Printing results
    print("Accuracy score: ", accuracy_score(test["class"], predictions))
    print("Confusion matrix: \n", confusion_matrix(test["class"], predictions))
    print("Classification report: \n", classification_report(test["class"], predictions))

    # Save the trained model
    with open(save_dir, "wb") as f:
        pickle.dump(model, f)

    return model

from sklearn.neural_network import MLPClassifier

def train_mlp(train, test, save_dir):
    train["text"].fillna("", inplace=True)
    test["text"].fillna("", inplace=True)

    # Data vectorizing
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])

    # Training model
    model = MLPClassifier(hidden_layer_sizes=(10), max_iter=100)
    model.fit(X_train, train["class"])

    # Predicting test set
    predictions = model.predict(X_test)

    # Printing results
    print("Accuracy score: ", accuracy_score(test["class"], predictions))
    print("Confusion matrix: \n", confusion_matrix(test["class"], predictions))
    print("Classification report: \n", classification_report(test["class"], predictions))

    # Save the trained model
    with open(save_dir, "wb") as f:
        pickle.dump(model, f)

    return model

from sklearn.ensemble import GradientBoostingClassifier

def train_gbm(train, test, save_dir):
    train["text"].fillna("", inplace=True)
    test["text"].fillna("", inplace=True)

    # Data vectorizing
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train["text"])
    X_test = vectorizer.transform(test["text"])

    # Training model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, train["class"])

    # Predicting test set
    predictions = model.predict(X_test)

    # Printing results
    print("Accuracy score: ", accuracy_score(test["class"], predictions))
    print("Confusion matrix: \n", confusion_matrix(test["class"], predictions))
    print("Classification report: \n", classification_report(test["class"], predictions))

    # Save the trained model
    with open(save_dir, "wb") as f:
        pickle.dump(model, f)

    return model

def check_model(input_text, model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict([input_text])

    if prediction == 0:
        print("Non-harmful")
    elif prediction == 1:
        print("Harmful")

    return prediction


def check_flair_model(input_text, best_model_path, final_model_path):
    # Load the best and final models
    best_model = TextClassifier.load(best_model_path)
    final_model = TextClassifier.load(final_model_path)

    # Create a Sentence object from the input text
    sentence = Sentence(input_text)

    # Use the best model to predict the label of the sentence
    best_model.predict(sentence)

    # Get the predicted label and confidence
    best_predicted_label = sentence.labels[0].value
    best_confidence = sentence.labels[0].score

    print(f"Best model prediction: Label - {best_predicted_label}, Confidence - {best_confidence}")

    # Use the final model to predict the label of the sentence
    final_model.predict(sentence)

    # Get the predicted label and confidence
    final_predicted_label = sentence.labels[0].value
    final_confidence = sentence.labels[0].score

    print(f"Final model prediction: Label - {final_predicted_label}, Confidence - {final_confidence}")

    # Return 'Harmful' or 'Non-harmful' based on the predicted label
    if final_predicted_label == '1':
        return 'Harmful'
    else:
        return 'Non-harmful'







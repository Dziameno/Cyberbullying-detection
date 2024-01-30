import os
import pickle

import numpy as np
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus, DataLoader
from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import class_weight
from datasets import load_dataset
from setfit import SetFitModel, Trainer, sample_dataset, TrainingArguments, losses
from sentence_transformers import SentenceTransformer, InputExample, losses
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader

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

    # Get the probabilities for each class
    probabilities = text_clf.decision_function(test["text"])

    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(test["class"], probabilities)

    # Compute the area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Extract the model name from the model path
    model_name = os.path.basename(save_dir)

    # Save the ROC curve to a file
    plt.savefig(f'../roc_curves/{model_name}.png')

    plt.show()

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
                         ('clf', LinearSVC()),
                         ])

    # Training model
    text_clf.fit(train["text"], train["class"])

    # Predicting test set
    predictions = text_clf.predict(test["text"])

    # Get the probabilities for each class
    probabilities = text_clf.decision_function(test["text"])

    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(test["class"], probabilities)

    # Compute the area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Extract the model name from the model path
    model_name = os.path.basename(save_dir)

    # Save the ROC curve to a file
    plt.savefig(f'../roc_curves/{model_name}.png')

    plt.show()

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

from datasets import Dataset

def setfit(model_link, save_dir):
    model = SetFitModel.from_pretrained(model_link)

    dataset = load_dataset("poleval2019_cyberbullying", "task01")

    train_dataset = sample_dataset(dataset["train"], label_column="label", num_samples=20)
    test_dataset = dataset["test"]

    model.labels = ["0", "1"]

    args = TrainingArguments(
        batch_size=128,
        num_epochs=10,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )

    trainer.train()

    trainer.evaluate(test_dataset)

    SetFitModel.save_pretrained(model, save_dir)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from transformers import AutoTokenizer

def sentence_transformer(model_link, save_dir):
    dataset = load_dataset("poleval2019_cyberbullying", "task01")

    model = SentenceTransformer(model_link)

    # Get the list of sentences
    texts = dataset["train"]["text"]

    # Convert the training data into the format required by SentenceTransformer
    # Create pairs of sentences for the CosineSimilarityLoss
    labels = dataset["train"]["label"]
    train_examples = [InputExample(texts=[texts[i], texts[i + 1]], label=float(labels[i])) for i in
                      range(0, len(texts) - 1, 2)]

    # Create a DataLoader for the training data
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Define the training procedure
    train_loss = losses.CosineSimilarityLoss(model)

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

    # Save the trained model
    model.save(save_dir)


# Multinomial Naive Bayes model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

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

    # Get the probabilities for each class
    probabilities = model.predict_proba(X_test)[:, 1]

    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(test["class"], probabilities)

    # Compute the area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Extract the model name from the model path
    model_name = os.path.basename(save_dir)

    # Save the ROC curve to a file
    plt.savefig(f'../roc_curves/{model_name}.png')

    plt.show()

    # Printing results
    print("Accuracy score: ", accuracy_score(test["class"], predictions))
    print("Confusion matrix: \n", confusion_matrix(test["class"], predictions))
    print("Classification report: \n", classification_report(test["class"], predictions))

    # Save the trained model
    with open(save_dir, "wb") as f:
        pickle.dump(model, f)

    # Save the vectorizer used during training
    with open('../models/vectorizers/mnb_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

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

    # Get the probabilities for each class
    probabilities = model.predict_proba(X_test)[:, 1]

    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(test["class"], probabilities)

    # Compute the area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Extract the model name from the model path
    model_name = os.path.basename(save_dir)

    # Save the ROC curve to a file
    plt.savefig(f'../roc_curves/{model_name}.png')

    plt.show()

    # Printing results
    print("Accuracy score: ", accuracy_score(test["class"], predictions))
    print("Confusion matrix: \n", confusion_matrix(test["class"], predictions))
    print("Classification report: \n", classification_report(test["class"], predictions))

    # Save the trained model
    with open(save_dir, "wb") as f:
        pickle.dump(model, f)

    # Save the vectorizer used during training
    with open('../models/vectorizers/mlp_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

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

    # Get the probabilities for each class
    probabilities = model.predict_proba(X_test)[:, 1]

    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(test["class"], probabilities)

    # Compute the area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Extract the model name from the model path
    model_name = os.path.basename(save_dir)

    # Save the ROC curve to a file
    plt.savefig(f'../roc_curves/{model_name}.png')

    plt.show()

    # Printing results
    print("Accuracy score: ", accuracy_score(test["class"], predictions))
    print("Confusion matrix: \n", confusion_matrix(test["class"], predictions))
    print("Classification report: \n", classification_report(test["class"], predictions))

    # Save the trained model
    with open(save_dir, "wb") as f:
        pickle.dump(model, f)

    # Save the vectorizer used during training
    with open('../models/vectorizers/gbm_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)

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

def check_model_proba(input_text, model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    prediction = model.decision_function([input_text])

    if prediction <= 0:
        print("Non-harmful" + " " + str(prediction))
    else:
        print("Harmful" + " " + str(prediction))

    return prediction

def check_other_models(input_text, model_path, vectorizer_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load the vectorizer used during training
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Transform the input text using the loaded vectorizer
    vft = vectorizer.transform([input_text])

    prediction = model.predict_proba(vft)[:, 1] * 100

    if prediction < 50:
        print("Non-harmful" + " " + str(prediction))
    else:
        print("Harmful" + " " + str(prediction))

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

def check_setfit_model(input_text, model_path):
    model = SetFitModel.from_pretrained(model_path)
    preds = model.predict([input_text])

    if preds == ['1']:
        print("Harmful")
    else:
        print("Non-harmful")


from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt

def matrix_report_hf(model_path, test):
    model = SetFitModel.from_pretrained(model_path)

    true_labels_list = []
    predicted_labels_list = []

    # Iterate over each row in the test set
    for index, row in test.iterrows():
        text = row["text"]

        # Get the predicted labels for the current text
        try:
            predictions = model.predict([text])
            predicted_label = str(predictions[0])
        except Exception as e:
            print(f"An error occurred during prediction for text '{text}': {e}")
            predicted_label = None

        # Append true and predicted labels to the lists
        true_labels_list.append(str(row["class"]))
        predicted_labels_list.append(predicted_label)

    # Convert the true labels and predicted labels to integers
    true_labels_list = [int(label) for label in true_labels_list]
    predicted_labels_list = [int(label) for label in predicted_labels_list]

    # Compute the false positive rate, true positive rate, and thresholds
    fpr, tpr, thresholds = roc_curve(true_labels_list, predicted_labels_list)

    # Compute the area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # Extract the model name from the model path
    model_name = os.path.basename(model_path)

    # Save the ROC curve to a file
    plt.savefig(f'../roc_curves/{model_name}.png')

    plt.show()

    # Using sklearn to print the confusion matrix and classification report
    print("Confusion Matrix:\n", confusion_matrix(true_labels_list, predicted_labels_list))
    print("Classification Report:\n", classification_report(true_labels_list, predicted_labels_list))



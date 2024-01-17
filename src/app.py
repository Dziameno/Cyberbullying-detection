import pandas as pd

from src.data.preprocess import load_dataset_from_disk, merge_data, preprocess_data
from src.models.models import train_svm, train_svm_balanced, check_svm, check_svm_balanced
from src.data.augmentation import back_translation

if __name__ == "__main__":
    # # Load data from disk
    # # Train set
    # train_text = load_dataset_from_disk("../data/Train/training_set_clean_only_text.txt")
    # train_tags = load_dataset_from_disk("../data/Train/training_set_clean_only_tags.txt")
    # # Test set
    # test_text = load_dataset_from_disk("../data/Test/test_set_clean_only_text.txt")
    # test_tags = load_dataset_from_disk("../data/Test/test_set_clean_only_tags.txt")

    # # Merging data into one csv file
    # merge_data(train_text, train_tags, "../data/Train/train_merged.csv")
    # merge_data(test_text, test_tags, "../data/Test/test_merged.csv")

    # # Preprocessing data
    # preprocess_data("../data/Train/train_merged.csv", "../data/Train/train_preprocessed.csv")
    # preprocess_data("../data/Test/test_merged.csv", "../data/Test/test_preprocessed.csv")

    # # Results after preprocessing
    # print("Train set classes distribution: ", train["class"].value_counts())
    # print("Test set classes distribution: ", test["class"].value_counts())

    # train = pd.read_csv("../data/Train/train_preprocessed.csv", sep="\t")
    # test = pd.read_csv("../data/Test/test_preprocessed.csv", sep="\t")
    #
    # # SVM model
    # train_svm_balanced(train, test, "../models/svm_model_balanced.pkl")
    # train_svm(train, test, "../models/svm_model.pkl")

    check_svm("Ty dzbanie")
    check_svm_balanced("Ty dzbanie")

    # #Augmentation
    #
    # # Back translation
    # back_translation("../data/Train/train_preprocessed.csv", "../data/Train/train_back_translation.csv")
    #
    # btran = pd.read_csv("../data/Train/train_back_translation.csv", sep="\t")
    #
    # # Results after back translation
    # train_svm(btran, test, "../models/svm_model_back_translation.pkl")
    # check_svm("")

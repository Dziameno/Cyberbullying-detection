from src.data.preprocess import load_dataset_from_disk, merge_data, preprocess_data
from src.models.models import train_svm, check_svm

if __name__ == "__main__":
    # # Load data from disk
    # # Train set
    train_text = load_dataset_from_disk("../data/Train/training_set_clean_only_text.txt")
    train_tags = load_dataset_from_disk("../data/Train/training_set_clean_only_tags.txt")
    # # Test set
    test_text = load_dataset_from_disk("../data/Test/test_set_clean_only_text.txt")
    test_tags = load_dataset_from_disk("../data/Test/test_set_clean_only_tags.txt")

    # # Merging data into one csv file
    merge_data(train_text, train_tags, "../data/Train/train_merged.csv")
    merge_data(test_text, test_tags, "../data/Test/test_merged.csv")

    # # Preprocessing data
    train = preprocess_data("../data/Train/train_merged.csv", "../data/Train/train_preprocessed.csv")
    test = preprocess_data("../data/Test/test_merged.csv", "../data/Test/test_preprocessed.csv")

    # # Results after preprocessing
    # print("Train set classes distribution: ", train["class"].value_counts())
    # print("Test set classes distribution: ", test["class"].value_counts())

    # #SVM model
    #train_svm(train, test, "../models/svm_model.pkl")
    check_svm("Masz dwie głowy jak hydra, jedna głupia, druga jeszcze głupsza")






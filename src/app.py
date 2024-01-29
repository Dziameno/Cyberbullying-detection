from random import sample

from src.data.preprocess import *
from src.models.models import *
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

    train = pd.read_csv("../data/Train/train_preprocessed.csv", sep="\t")
    test = pd.read_csv("../data/Test/test_preprocessed.csv", sep="\t")
    
    # # SVM model
    # train_svm_balanced(train, test, "../models/svm_model_balanced.pkl")
    # train_svm(train, test, "../models/svm_model.pkl")

    # Multinominal Naive Bayes model
    # train_mnb(train, test, "../models/mnb_model.pkl")
    
    # Multilayer Perceptron model
    # train_mlp(train, test, "../models/mlp_model.pkl")
    
    # Gradient Boosting Machines model
    # train_gbm(train, test, "../models/gbm_model.pkl")


    # # Flair model
    # train_file = "../data/Train/train_preprocessed.csv"
    # test_file = "../data/Test/test_preprocessed.csv"
    #
    # train_flair_classifier(train_file, test_file, "../models/flair_model")

    # SetFit model
    # setfit("OrlikB/st-polish-kartonberta-base-alpha-v1", "../models/st-polish-kartonberta-base-alpha-v1")
    # matrix_report_hf("../models/st-polish-kartonberta-base-alpha-v1", test)

    # check_model("Ty dzbanie", "../models/svm_model.pkl")
    # check_model("Ty dzbanie", "../models/svm_model_balanced.pkl")
    # check_flair_model("Ty dzbanie", "../models/flair_model/best-model.pt", "../models/flair_model/final-model.pt")
    # check_setfit_model("Ty dzbanie", "../models/st-polish-kartonberta-base-alpha-v1")

    # check_model("Ty dzbanie", "../models/mnb_model.pkl")
    # check_model("Ty dzbanie", "../models/mlp_model.pkl")
    # check_model("Ty dzbanie", "../models/gbm_model.pkl")

    # #Augmentation
    # # Back translation
    # back_translation("../data/Train/train_preprocessed.csv", "../data/Train/train_back_translation.csv")
    #
    # btran = pd.read_csv("../data/Train/train_back_translation.csv", sep="\t")
    #
    # # Results after back translation
    # train_svm(btran, test, "../models/svm_model_back_translation.pkl")
    # check_svm("")

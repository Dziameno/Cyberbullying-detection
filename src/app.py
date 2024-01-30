from src.data.preprocess import *
from src.models.models import *
from src.data.augmentation import back_translation
from src.data.confusion_matrix import plot_matrix

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
    # train_svm(train, test, "../models/svm_model_proba.pkl")
    # train_svm_balanced(train, test, "../models/svm_model_balanced_proba.pkl")


    # Multinominal Naive Bayes model
    # train_mnb(train, test, "../models/mnb_model_proba.pkl")
    
    # Multilayer Perceptron model
    # train_mlp(train, test, "../models/mlp_model_proba.pkl")
    
    # Gradient Boosting Machines model
    # train_gbm(train, test, "../models/gbm_model_proba.pkl")


    # # Flair model
    # train_file = "../data/Train/train_preprocessed.csv"
    # test_file = "../data/Test/test_preprocessed.csv"
    #
    # train_flair_classifier(train_file, test_file, "../models/flair_model")


    # SetFit model
    # setfit("OrlikB/st-polish-kartonberta-base-alpha-v1", "../models/st-polish-kartonberta-base-alpha-v1")
    # matrix_report_hf("../models/st-polish-kartonberta-base-alpha-v1", test)
    # setfit("sdadas/st-polish-paraphrase-from-mpnet", "../models/st-polish-paraphrase-from-mpnet")
    # matrix_report_hf("../models/st-polish-paraphrase-from-mpnet", test)
    # setfit("sdadas/mmlw-roberta-base", "../models/mmlw-roberta-base")
    # matrix_report_hf("../models/mmlw-roberta-base", test)
    # setfit("sdadas/mmlw-roberta-large", "../models/mmlw-roberta-large")
    # matrix_report_hf("../models/mmlw-roberta-large", test)


    # Sentence Transformers
    # sentence_transformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "../models/paraphrase-multilingual-MiniLM-L12-v2")
    # matrix_report_hf("../models/paraphrase-multilingual-MiniLM-L12-v2", test)
    # sentence_transformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "../models/paraphrase-multilingual-mpnet-base-v2")
    # matrix_report_hf("../models/paraphrase-multilingual-mpnet-base-v2", test)
    # sentence_transformer("sentence-transformers/distiluse-base-multilingual-cased-v1", "../models/distiluse-base-multilingual-cased-v1")

    # check_model("Ty dzbanie", "../models/svm_model.pkl")
    # check_model_proba("Ty dzbanie", "../models/svm_model_proba.pkl")
    #
    # check_model("Ty dzbanie", "../models/svm_model_balanced.pkl")
    # check_model_proba("Ty dzbanie", "../models/svm_model_balanced_proba.pkl")

    # check_other_models("Ty dzbanie", "../models/mnb_model_proba.pkl", "../models/vectorizers/mnb_vectorizer.pkl")
    # check_other_models("Ty dzbanie", "../models/mlp_model_proba.pkl","../models/vectorizers/mlp_vectorizer.pkl")
    # check_other_models("Ty dzbanie", "../models/gbm_model_proba.pkl","../models/vectorizers/gbm_vectorizer.pkl")

    # # check_flair_model("Ty dzbanie", "../models/flair_model/best-model.pt", "../models/flair_model/final-model.pt")
    # check_setfit_model("Ty człowieku", "../models/st-polish-kartonberta-base-alpha-v1")
    # check_setfit_model("Ty człowieku", "../models/st-polish-paraphrase-from-mpnet")
    # check_setfit_model("Ty człowieku", "../models/mmlw-roberta-base")
    # check_setfit_model("Ty człowieku", "../models/mmlw-roberta-large")

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

    # Ploting confusion matrixes for models
    # plot_matrix(816,5,100,21, "../matrixes/svc.png")
    # plot_matrix(795,26,74,47, "../matrixes/svc_balanced.png")
    # plot_matrix(818,3,111,10, "../matrixes/mnb.png")
    # plot_matrix(802,19,94,27, "../matrixes/mp.png")
    # plot_matrix(821,0,114,7, "../matrixes/gbm.png")
    # plot_matrix(680,141,29,92, "../matrixes/st-polish-kartonberta-base-alpha-v1_20.png")
    # plot_matrix(660,161,20,101, "../matrixes/st-polish-kartonberta-base-alpha-v1_50.png")
    # plot_matrix(621,200,26,95, "../matrixes/st-polish-paraphrase-from-mpnet.png")
    # plot_matrix(639,182,21,100, "../matrixes/mmlw-roberta-base.png")
    # plot_matrix(586,235,18,103, "../matrixes/mmlw-roberta-large.png")
    # plot_matrix(24,797,0,121, "../matrixes/paraphrase-multilingual-MiniLM-L12-v2.png")

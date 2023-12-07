from datasets import load_dataset
import pandas as pd


# # Loading dataset from HuggingFace Datasets
# # Choose dataset based on task
# # Task 1: Harmful (class: 1) vs Non-harmful (class: 0)
# # Task 2: Type of harmfulness (class: 0 - non-harmful, 1 - cyberbullying, 2 - hate speech)
# def load_dataset_from_huggingface(dataset_name, task_name):
#     dataset = load_dataset(dataset_name, task_name)
#
#     save_dir = "data_from_huggingface"
#     dataset.save_to_disk(save_dir)
#
#     return dataset
#
#
# load_dataset_from_huggingface("poleval2019_cyberbullying", "task01")


# # Loading dataset downloaded from official website
# # http://2019.poleval.pl/index.php/tasks/task6
def load_dataset_from_disk(dataset_dir):

    return pd.read_csv(dataset_dir, sep="\t")


# # Initial merging of data
def merge_data(only_text, tags_classes, save_dir):
    # Adding id column to only_text and only_tags
    only_text["id"] = only_text.index
    tags_classes["id"] = tags_classes.index

    # Merge dataframes based on id column
    merged = pd.merge(only_text, tags_classes, on="id")

    # Delete id columns needed for merging
    del merged["id"]

    # Add column names - text and class
    merged.columns = ["text", "class"]

    # Saving merged dataset to csv file
    merged.to_csv(save_dir, sep="\t", index=False)

    return merged


# # Preprocessing dataset (cleaning text)
def preprocess(dataset_dir, new_dataset_dir):
    # Load data from disk
    dataset = load_dataset_from_disk(dataset_dir)
    # Dataframe
    df = pd.DataFrame(dataset)
    # Remove duplicates with RT (retweet)
    df = df[~df["text"].str.contains("RT")]
    # Remove duplicates with the same text
    df = df.drop_duplicates(subset=["text"])
    # Remove urls
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)
    # Remove html tags
    df["text"] = df["text"].str.replace(r"<.*?>", "", regex=True)
    # Remove @ with names of users standing after it
    df["text"] = df["text"].str.replace(r'@\w+', "", regex=True)
    # Remove # with words standing after it
    df["text"] = df["text"].str.replace(r"#\S+", "", regex=True)
    # Remove whitespaces
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
    # Remove spaces and double spaces at the beginning and at the end of the text
    df["text"] = df["text"].str.strip()
    # Remove punctuation
    df["text"] = df["text"].str.replace(r"[^\w\s]", "", regex=True)
    # Remove numbers
    df["text"] = df["text"].str.replace(r"\d+", "", regex=True)
    # Remove emojis and letters after them
    df["text"] = df["text"].str.replace(r"[\U0001F600-\U0001F64F]+", "", regex=True)
    # Remove smileys like :) or :D or :P etc.
    df["text"] = df["text"].str.replace(r"[:;=][oO\-]?[D\)\]\(\]/\\OpP]", "", regex=True)
    # Remove reserved words
    df["text"] = df["text"].str.replace(r"RT", "", regex=True)
    # Remove polish emojis like XD,xD,xd, xdd, x, xx etc.
    df["text"] = df["text"].str.replace(r"[xX][dD]+", "", regex=True)
    # Save preprocessed dataset to disk
    df.to_csv(new_dataset_dir, sep="\t", index=False)

    return df




# # Load data from disk
# # Train set
train_text = load_dataset_from_disk("data/Train/training_set_clean_only_text.txt")
train_tags = load_dataset_from_disk("data/Train/training_set_clean_only_tags.txt")
# # Test set
test_text = load_dataset_from_disk("data/Test/test_set_clean_only_text.txt")
test_tags = load_dataset_from_disk("data/Test/test_set_clean_only_tags.txt")

# # Merging data into one csv file
merge_data(train_text, train_tags, "data/Train/train_merged.csv")
merge_data(test_text, test_tags, "data/Test/test_merged.csv")

# # Preprocessing data
train = preprocess("data/Train/train_merged.csv", "data/Train/train__preprocessed.csv")
test = preprocess("data/Test/test_merged.csv", "data/Test/test_preprocessed.csv")

# #Results after preprocessing
# print("Train set classes distribution: ", train["class"].value_counts())
# print("Test set classes distribution: ", test["class"].value_counts())














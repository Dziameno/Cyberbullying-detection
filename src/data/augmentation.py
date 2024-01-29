from src.data.preprocess import load_dataset_from_disk
from difflib import SequenceMatcher
import pandas as pd
from googletrans import Translator

def back_translation(dataset_dir, new_dataset_dir):
    # Load data from disk
    dataset = load_dataset_from_disk(dataset_dir)
    # Dataframe
    df = pd.DataFrame(dataset)

    translator = Translator()

    with open(new_dataset_dir, 'w', encoding="utf-8") as f:
        f.write("text\tclass\n")
        for index, row in df.iterrows():
            # Check if the class is 1
            if row.iloc[1] == 1:  # Use iloc to access values by position
                try:
                    # Translating first column to English
                    translation = translator.translate(str(row.iloc[0]), dest='en')
                    # Translating first column back to Polish
                    translation = translator.translate(translation.text, dest='pl')

                    f.write(translation.text + "\t" + str(row.iloc[1]) + "\n")

                    if index % 100 == 0:
                        print(f"Translated {index} sentences.")

                except Exception as e:
                    # Handle translation failure by using the original sentence
                    print(f"Translation failed for index {index}. Using the original sentence.")
                    f.write(str(row.iloc[0]) + "\t" + str(row.iloc[1]) + "\n")
            else:
                # If class is not 1, use the original sentence without translation
                f.write(str(row.iloc[0]) + "\t" + str(row.iloc[1]) + "\n")

    return df

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def back_translation(dataset_dir, new_dataset_dir):
    # Load data from disk
    dataset = load_dataset_from_disk(dataset_dir)
    # Dataframe
    df = pd.DataFrame(dataset)

    translator = Translator()

    with open(new_dataset_dir, 'w', encoding="utf-8") as f:
        f.write("text\tclass\n")
        for index, row in df.iterrows():
            original_sentence = str(row.iloc[0])
            # Check if the class is 1
            if row.iloc[1] == 1:  # Use iloc to access values by position
                try:
                    # Back Translation
                    translation = translator.translate(original_sentence, dest='en').text
                    translation_back = translator.translate(translation, dest='pl').text

                    # Compare with the original sentence, if similarity is below a threshold, add to the new dataset
                    if similarity(translation_back.lower(), original_sentence.lower()) < 0.9:
                        f.write(translation_back + "\t" + str(row.iloc[1]) + "\n")
                        if index % 100 == 0:
                            print(f"Back-Translated {index} sentences.")
                    else:
                        # If the similarity is high, use the original sentence
                        f.write(original_sentence + "\t" + str(row.iloc[1]) + "\n")

                except Exception as e:
                    # Handle translation failure by using the original sentence
                    # print(f"Translation failed for index {index}. Using the original sentence.")
                    f.write(original_sentence + "\t" + str(row.iloc[1]) + "\n")
            else:
                # If class is not 1, use the original sentence without translation
                f.write(original_sentence + "\t" + str(row.iloc[1]) + "\n")

    return df
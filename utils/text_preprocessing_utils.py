import csv
import re
import sys

# Function to clean and lowercase text
def preprocess_text(text):
    # Converts to lowercase
    text = text.lower()
    # Replaces encoded single quotes/apostrophes and other symbols
    text = text.replace("â\x80\x99", "'")  # Replaces with regular apostrophe
    text = text.replace("â\x80\x9c", "")   # Removes opening quotation-like symbol
    text = text.replace("â\x80\x9d", "")   # Removes closing quotation-like symbol
    text = text.replace("â\x80¦", "...")   # Replacs with ellipsis
    text = text.replace("\\'", "'")        # Replaces backslash before single quotes
    # Removes any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

# Function to load and preprocess the dataset

def load_and_preprocess_dataset(csv_file, text_column):
    events = []
    csv.field_size_limit(sys.maxsize)  # Set the field size limit to maximum

    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skips header
        for row in reader:
            # Apply any preprocessing steps here (like text cleaning, removing unwanted characters, etc.)
            processed_text = preprocess_text(row[text_column]) 
            events.append({"text": processed_text})

    return events
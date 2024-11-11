import csv
import re

# Function to clean and lowercase text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Replace encoded single quotes/apostrophes and other symbols
    text = text.replace("â\x80\x99", "'")  # Replace with regular apostrophe
    text = text.replace("â\x80\x9c", "")   # Remove opening quotation-like symbol
    text = text.replace("â\x80\x9d", "")   # Remove closing quotation-like symbol
    text = text.replace("â\x80¦", "...")   # Replace with ellipsis
    text = text.replace("\\'", "'")        # Replace backslash before single quotes
    # Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

# Function to load and preprocess the dataset
def load_and_preprocess_dataset(csv_file, text_column):
    events = []
    max_size = csv.field_size_limit()
    csv.field_size_limit(2 * max_size)  # Double the limit

    with open(csv_file, 'r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header

        for row in reader:
            # Clean and add the processed text
            processed_text = preprocess_text(row[text_column])
            events.append({"text": processed_text})

    return events
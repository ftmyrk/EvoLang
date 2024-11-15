import pandas as pd

# Load the dataset from the given path
old_event_csv = '/home/otamy001/EvoLang/generated_responses_2013.csv'

# Read the first 20 rows of the dataset
first_20_rows = pd.read_csv(old_event_csv, nrows=20)
print(first_20_rows)
import pandas as pd
import os

input_file = '/home/otamy001/EvoLang/Dataset/abcnews-date-text.csv'   
output_directory = '/home/otamy001/EvoLang/Dataset/YearlyFiles/'  
os.makedirs(output_directory, exist_ok=True)

df = pd.read_csv(input_file, names=["Date", "Headline"])   

df['Year'] = df['Date'].astype(str).str[:4]

for year, group in df.groupby('Year'):
    output_file = os.path.join(output_directory, f"{year}_year.csv")
    group.drop(columns=["Year"], inplace=True)  # Remove the Year column before saving
    group.to_csv(output_file, index=False, header=False)  # Save without index and header
    print(f"Saved: {output_file}")

# # Group the dataset by Year and save each group to a separate file
# for year, group in df.groupby('Year'):
#     output_file = os.path.join(output_directory, f"{year}_year.csv")
#     group.drop(columns=["Year"], inplace=True)  # Remove the Year column before saving
#     group.to_csv(output_file, index=False, header=False)
#     print(f"Saved: {output_file}")
# def convert_date_format(date_str):
#     # Adjust the format to match the input format: day, month, year
#     date_obj = datetime.strptime(date_str, '%d%m%Y')
#     # Return the formatted string
#     return date_obj.strftime('%d %B %Y')

# # Example usage
# date_str = ""  # 8th December 2023
# formatted_date = convert_date_format(date_str)
# print(formatted_date)  # Output: "08 December 2023"# print(date) 
# dataset_path = "/home/otamy001/EvoLang/generated_data/generated_responses_2023.csv"
# output_path = "/home/otamy001/EvoLang/generated_data/generated_responses_2013.csv"

# df1 = pd.read_csv(dataset_path)
# df2 = pd.read_csv(output_path)

# print(len(df1))
# print(len(df2))

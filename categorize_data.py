import pandas as pd
import os

categories = {
    "Health": ["vaccine", "COVID-19", "pandemic", "lockdown", "hospital", "virus"],
    "Technology": ["AI", "artificial intelligence", "smartphone", "tablet", "robot", "5G"],
    "Economy": ["stocks", "market", "economy", "inflation", "finance", "GDP"],
    "Politics": ["election", "policy", "government", "senate", "president"],
    "Sports": ["soccer", "football", "Olympics", "basketball", "cricket"]
}

def assign_category(headline):
    for category, keywords in categories.items():
        if any(keyword.lower() in headline.lower() for keyword in keywords):
            return category
    return "Other"

def categorize_dataset(input_file, output_dir, year):
    print(f"Categorizing data from {input_file}...")
    df = pd.read_csv(input_file, names=["Date", "Headline"])
    df["Category"] = df["Headline"].apply(assign_category)
    
    for category in df["Category"].unique():
        category_df = df[df["Category"] == category]
        output_path = os.path.join(output_dir, f"{year}_{category}_categorized.csv")
        category_df.to_csv(output_path, index=False)
        print(f"Saved categorized data for {category} to {output_path}")

input_2011 = "/home/otamy001/EvoLang/Dataset/YearlyFiles/2011_year.csv"
input_2021 = "/home/otamy001/EvoLang/Dataset/YearlyFiles/2021_year.csv"
output_dir = "/home/otamy001/EvoLang/generated_data/category_related/"
os.makedirs(output_dir, exist_ok=True)

categorize_dataset(input_2011, output_dir, 2011)
categorize_dataset(input_2021, output_dir, 2021)
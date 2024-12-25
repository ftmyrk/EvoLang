import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils.macros import OUTPUT_DIR
import os

filtered_2013 = pd.read_csv(f"{OUTPUT_DIR}/filtered_2013.csv")
filtered_2023 = pd.read_csv(f"{OUTPUT_DIR}/filtered_2023.csv")
dictionary_analysis_2013 = pd.read_csv(f"{OUTPUT_DIR}/dictionary_analysis_2013.csv")
dictionary_analysis_2023 = pd.read_csv(f"{OUTPUT_DIR}/dictionary_analysis_2023.csv")

def analyze_by_category(filtered_data, dictionary_data, year, output_dir):
    merged_data = filtered_data.merge(dictionary_data, left_index=True, right_index=True)

    category_analysis = merged_data.groupby("Category").mean()

    output_file = os.path.join(output_dir, f"category_analysis_{year}.csv")
    category_analysis.to_csv(output_file)
    print(f"Category analysis for {year} saved to {output_file}")

    plt.figure(figsize=(12, 8))
    category_analysis.plot(kind="bar", figsize=(12, 8))
    plt.title(f"Stylistic Trends by Category ({year})")
    plt.ylabel("Average Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"category_trends_{year}.png"))
    plt.close()
    print(f"Category trends visualization for {year} saved.")

analyze_by_category(filtered_2013, dictionary_analysis_2013, 2013, OUTPUT_DIR)
analyze_by_category(filtered_2023, dictionary_analysis_2023, 2023, OUTPUT_DIR)

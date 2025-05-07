import pandas as pd
import ast
import os

# Ensure the output directory exists
output_dir = "normalized_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the CSV file
df = pd.read_csv('./raw_data/keywords.csv')  # Update the path as necessary

# Function to extract and normalize keywords
def normalize_keywords(keyword_str):
    if pd.isna(keyword_str) or not isinstance(keyword_str, str):  # Check if the value is NaN or not a string
        return []
    # Convert the stringified JSON object into a list of dictionaries
    keyword_list = ast.literal_eval(keyword_str)
    # Extract the 'name' of each keyword and return them as a list
    return [kw['name'] for kw in keyword_list]

# Apply the normalization function to the 'keywords' column
df['keywords'] = df['keywords'].apply(normalize_keywords)

# Combine all keywords for each tmdbId into a single row
df_keywords_aggregated = df.groupby('id', as_index=False).agg({'keywords': lambda x: ', '.join(sum(x, []))})

# Rename the 'id' column to 'tmdbId'
df_keywords_aggregated.rename(columns={'id': 'tmdbId'}, inplace=True)

# Save the aggregated DataFrame to a new CSV file
df_keywords_aggregated.to_csv(os.path.join(output_dir, 'normalized_keywords.csv'), index=False)

# Display the first few rows of the aggregated DataFrame for verification
print(df_keywords_aggregated.head())

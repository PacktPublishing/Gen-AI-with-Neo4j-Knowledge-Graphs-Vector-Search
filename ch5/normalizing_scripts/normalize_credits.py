import pandas as pd
import ast
import os

# Ensure the output directory exists
output_dir = "normalized_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the CSV file
df = pd.read_csv('./raw_data/credits.csv')

# Function to extract relevant cast information
def extract_cast(cast_str):
    cast_list = ast.literal_eval(cast_str)
    return [{'actor_id': c['id'], 'name': c['name'], 'character': c['character'], 'cast_id': c['cast_id']} for c in cast_list]

# Function to extract relevant crew information
def extract_crew(crew_str):
    crew_list = ast.literal_eval(crew_str)
    relevant_jobs = ['Director', 'Producer']
    return [{'crew_id': c['id'], 'name': c['name'], 'job': c['job']} for c in crew_list if c['job'] in relevant_jobs]

# Apply the extraction functions to each row
df['cast'] = df['cast'].apply(extract_cast)
df['crew'] = df['crew'].apply(extract_crew)

# Explode the lists into separate rows
df_cast = df.explode('cast').dropna(subset=['cast'])
df_crew = df.explode('crew').dropna(subset=['crew'])

# Normalize the exploded data
df_cast_normalized = pd.json_normalize(df_cast['cast'])
df_crew_normalized = pd.json_normalize(df_crew['crew'])

# Reset index to avoid duplicate indices
df_cast_normalized = df_cast_normalized.reset_index(drop=True)
df_crew_normalized = df_crew_normalized.reset_index(drop=True)

# Drop duplicate rows if any
df_cast_normalized = df_cast_normalized.drop_duplicates()
df_crew_normalized = df_crew_normalized.drop_duplicates()

# Add the movie ID back to the normalized DataFrames
df_cast_normalized['tmdbId'] = df_cast.reset_index(drop=True)['id']
df_crew_normalized['tmdbId'] = df_crew.reset_index(drop=True)['id']

# Save the normalized data with the updated column names
df_cast_normalized.to_csv(os.path.join(output_dir, 'normalized_cast.csv'), index=False)
df_crew_normalized.to_csv(os.path.join(output_dir, 'normalized_crew.csv'), index=False)

# Display a sample of the output for verification
print("Sample of normalized cast data:")
print(df_cast_normalized.head())

print("Sample of normalized crew data:")
print(df_crew_normalized.head())

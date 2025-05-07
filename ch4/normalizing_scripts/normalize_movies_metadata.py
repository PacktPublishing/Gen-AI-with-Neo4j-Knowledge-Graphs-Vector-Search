import pandas as pd
import ast
import os


# Ensure the output directory exists
output_dir = "normalized_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the CSV file
df = pd.read_csv('./raw_data/movies_metadata.csv')  # Update the path as necessary

# Function to extract and normalize genres
def extract_genres(genres_str):
    if pd.isna(genres_str) or not isinstance(genres_str, str):
        return []
    genres_list = ast.literal_eval(genres_str)
    return [{'genre_id': int(g['id']), 'genre_name': g['name']} for g in genres_list]

# Function to extract and normalize production companies
def extract_production_companies(companies_str):
    if pd.isna(companies_str) or not isinstance(companies_str, str):
        return []
    companies_list = ast.literal_eval(companies_str)
    if isinstance(companies_list, list):
        return [{'company_id': int(c['id']), 'company_name': c['name']} for c in companies_list]
    return []

# Function to extract and normalize production countries
def extract_production_countries(countries_str):
    if pd.isna(countries_str) or not isinstance(countries_str, str):
        return []
    countries_list = ast.literal_eval(countries_str)
    if isinstance(countries_list, list):
        return [{'country_code': c['iso_3166_1'], 'country_name': c['name']} for c in countries_list]
    return []

# Function to extract and normalize spoken languages
def extract_spoken_languages(languages_str):
    if pd.isna(languages_str) or not isinstance(languages_str, str):
        return []
    languages_list = ast.literal_eval(languages_str)
    if isinstance(languages_list, list):
        return [{'language_code': l['iso_639_1'], 'language_name': l['name']} for l in languages_list]
    return []

# Apply the extraction functions to each row
df['genres'] = df['genres'].apply(extract_genres)
df['production_companies'] = df['production_companies'].apply(extract_production_companies)
df['production_countries'] = df['production_countries'].apply(extract_production_countries)
df['spoken_languages'] = df['spoken_languages'].apply(extract_spoken_languages)

# Explode the lists into separate rows
df_genres = df.explode('genres').dropna(subset=['genres'])
df_companies = df.explode('production_companies').dropna(subset=['production_companies'])
df_countries = df.explode('production_countries').dropna(subset=['production_countries'])
df_languages = df.explode('spoken_languages').dropna(subset=['spoken_languages'])

# Normalize the exploded data
df_genres_normalized = pd.json_normalize(df_genres['genres'])
df_companies_normalized = pd.json_normalize(df_companies['production_companies'])
df_countries_normalized = pd.json_normalize(df_countries['production_countries'])
df_languages_normalized = pd.json_normalize(df_languages['spoken_languages'])

# Reset index to avoid duplicate indices
df_genres_normalized = df_genres_normalized.reset_index(drop=True)
df_companies_normalized = df_companies_normalized.reset_index(drop=True)
df_countries_normalized = df_countries_normalized.reset_index(drop=True)
df_languages_normalized = df_languages_normalized.reset_index(drop=True)

# Add the movie ID back to the normalized DataFrames as 'tmdbId'
df_genres_normalized['tmdbId'] = df_genres.reset_index(drop=True)['id']
df_companies_normalized['tmdbId'] = df_companies.reset_index(drop=True)['id']
df_countries_normalized['tmdbId'] = df_countries.reset_index(drop=True)['id']
df_languages_normalized['tmdbId'] = df_languages.reset_index(drop=True)['id']

# Ensure that 'company_id' and similar fields are treated as integers
df_companies_normalized['company_id'] = df_companies_normalized['company_id'].astype(int)
df_genres_normalized['genre_id'] = df_genres_normalized['genre_id'].astype(int)

# Save the normalized data with the updated column names
df_genres_normalized.to_csv(os.path.join(output_dir, 'normalized_genres.csv'), index=False)
df_companies_normalized.to_csv(os.path.join(output_dir, 'normalized_production_companies.csv'), index=False)
df_countries_normalized.to_csv(os.path.join(output_dir, 'normalized_production_countries.csv'), index=False)
df_languages_normalized.to_csv(os.path.join(output_dir, 'normalized_spoken_languages.csv'), index=False)

# For the movies, including "Belongs to Collection" within the same CSV
# Extract only the "name" from "belongs_to_collection" and include additional fields
def extract_collection_name(collection_str):
    if isinstance(collection_str, str):
        try:
            collection_dict = ast.literal_eval(collection_str)
            if isinstance(collection_dict, dict):
                return collection_dict.get('name', "None")
        except (ValueError, SyntaxError):  # Handle cases where string parsing fails
            return "None"
    return "None"

df_movies = df[['id', 'original_title', 'adult', 'budget', 'imdb_id', 'original_language', 'revenue', 'tagline', 'title', 'release_date', 'runtime', 'overview', 'belongs_to_collection']].copy()

df_movies['belongs_to_collection'] = df_movies['belongs_to_collection'].apply(extract_collection_name)
df_movies['adult'] = df_movies['adult'].apply(lambda x: 1 if x == 'TRUE' else 0)  # Convert 'adult' to integer
df_movies.rename(columns={'id': 'tmdbId'}, inplace=True)  # Rename 'id' to 'tmdbId'

# Save the movies to a separate CSV, including the extracted fields
df_movies.to_csv('./normalized_data/normalized_movies.csv', index=False)

# Display a sample of the output for verification
print("Sample of normalized genres data:")
print(df_genres_normalized.head())

print("Sample of normalized production companies data:")
print(df_companies_normalized.head())

print("Sample of normalized production countries data:")
print(df_countries_normalized.head())

print("Sample of normalized spoken languages data:")
print(df_languages_normalized.head())

print("Sample of movies data:")
print(df_movies.head())

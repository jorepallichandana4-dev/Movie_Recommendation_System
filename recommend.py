

import pandas as pd

# Load movie and ratings datasets
movies = pd.read_csv("dataset/movies.csv")
ratings = pd.read_csv("dataset/ratings.csv")

# Display first five movie records
print("MOVIE DATA:")
print(movies.head())

# Display first five rating records
print("\nRATINGS DATA:")
print(ratings.head())

# Display dataset information
print("\nTotal Movies:", len(movies))
print("Total Ratings:", len(ratings))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Replace missing genres with empty text
movies["genres"] = movies["genres"].fillna("")

# Convert movie genres into numerical values
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split("|"), token_pattern=None)

genre_matrix = vectorizer.fit_transform(movies["genres"])

# Calculate similarity between movies
similarity = cosine_similarity(genre_matrix)


# Function to recommend similar movies
def recommend_movies(movie_name, number=5):

    # Find the selected movie
    matches = movies[
        movies["title"].str.lower() == movie_name.lower()
    ]

    if matches.empty:
        print("Movie not found. Please enter an exact movie title.")
        return

    movie_index = matches.index[0]

    # Get similarity scores
    scores = list(enumerate(similarity[movie_index]))

    # Sort movies by similarity
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Display top recommendations (excluding selected movie)
    print("\nMovies recommended for you:")

    count = 0

    for index, score in scores[1:]:
        print(movies.iloc[index]["title"])
        count += 1

        if count == number:
            break


# Test the recommendation system
recommend_movies("Toy Story (1995)")

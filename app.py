
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("dataset/movies.csv")

# Clean genres
movies["genres"] = movies["genres"].fillna("")

# Convert genres into numerical values
vectorizer = TfidfVectorizer(
    tokenizer=lambda x: x.split("|"),
    token_pattern=None
)

genre_matrix = vectorizer.fit_transform(movies["genres"])

# Calculate similarity
similarity = cosine_similarity(genre_matrix)

# Streamlit page
st.set_page_config(page_title="Movie Recommendation System",
                   page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.write("Select a movie to discover similar movies.")

# Movie selection
movie_name = st.selectbox(
    "Choose a movie:",
    movies["title"].sort_values().tolist()
)

if st.button("Recommend Movies"):

    movie_index = movies[
        movies["title"] == movie_name
    ].index[0]

    scores = list(enumerate(similarity[movie_index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    st.subheader("Recommended Movies")
    for index, score in scores[1:6]:

        movie = movies.iloc[index]

        with st.container(border=True):

            st.subheader("🎬 " + movie["title"])

            st.write("**Genres:**", movie["genres"])

            st.progress(float(score))

            st.caption(
                f"Genre similarity: {score * 100:.2f}%"
            )

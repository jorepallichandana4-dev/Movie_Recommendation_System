# Movie Recommendation System

## Project Overview
The Movie Recommendation System is a Python-based application that recommends similar movies based on the genres of a movie selected by the user.

It uses Natural Language Processing (NLP) techniques and cosine similarity to identify movies with similar genres.

## Features
- Load movie and ratings datasets.
- Process movie genre information.
- Convert genres into numerical features using TF-IDF.
- Calculate movie similarity using cosine similarity.
- Recommend the top 5 similar movies.
- Interactive web application using Streamlit.
- Display movie genres and similarity percentages.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Dataset
This project uses the MovieLens Latest Small dataset.

Dataset source: https://grouplens.org/datasets/movielens/latest/

The dataset includes:
- movies.csv
- ratings.csv

## Project Structure
```text
Movie_Recommendation_System/
│
├── dataset/
│   ├── movies.csv
│   └── ratings.csv
│
├── recommend.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## Run the Project

Run the recommendation program:

```bash
python recommend.py
```

Start the web application:

```bash
streamlit run app.py
```

## How It Works
1. Load movie information from the dataset.
2. Extract movie genres.
3. Convert genres into numerical features using TF-IDF.
4. Calculate similarity using cosine similarity.
5. Find and display the top 5 similar movies.

## Author
Chandana Jorepalli
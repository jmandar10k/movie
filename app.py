import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from nltk.stem.porter import PorterStemmer

# Load data
movies = pd.read_csv("C:\\Users\\jmand\\OneDrive\\Desktop\\vitiligo\\movies\\tmdb_5000_movies.csv")
credits = pd.read_csv("C:\\Users\\jmand\\OneDrive\\Desktop\\vitiligo\\movies\\tmdb_5000_credits.csv")

# Merge movies and credits datasets on title
movies = movies.merge(credits, on='title')

# Keep relevant columns
movies = movies[['movie_id', 'genres', 'keywords', 'title', 'overview', 'cast', 'crew']]
movies.dropna(inplace=True)  # Remove missing values

# Helper functions for data preprocessing
def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l

def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L

def convert1(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l

# Data preprocessing
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['crew'] = movies['crew'].apply(convert1)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces from genres, keywords, cast, and crew
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

# Create tags column
movies['tags'] = movies['genres'] + movies['keywords'] + movies['overview'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())

# Vectorization and similarity calculation
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie):
    if movie not in movies['title'].values:
        return ["Movie not found."]
    
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    
    recommended_movies = [movies.iloc[i[0]].title for i in movie_list]
    return recommended_movies

# Streamlit UI
st.title('Movie Recommender System')

selected_movie = st.selectbox(
    'Select or type a movie to get recommendations:',
    movies['title'].values
)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.write('Recommended Movies:')
    for movie in recommendations:
        st.write(movie)

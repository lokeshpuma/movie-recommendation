import streamlit as st
import pandas as pd
import pickle
import numpy as np

MODEL_PATH = "movie_model.pkl"
MOVIES_PATH = "movies.csv"

@st.cache_data
def load_movies(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "title" not in df.columns:
        raise ValueError("movies.csv must contain a 'title' column")
    return df

@st.cache_data
def load_similarity_matrix(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        similarity = pickle.load(f)
    if not isinstance(similarity, np.ndarray):
        raise ValueError("The model file must contain a numpy array similarity matrix")
    return similarity

@st.cache_data
def get_title_to_index(df: pd.DataFrame) -> dict:
    return {title: idx for idx, title in enumerate(df["title"].tolist())}

@st.cache_data
def get_recommendations(title: str, df: pd.DataFrame, similarity: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    title_to_index = get_title_to_index(df)
    if title not in title_to_index:
        raise ValueError(f"Movie title '{title}' not found in dataset")

    idx = title_to_index[title]
    scores = similarity[idx]
    top_indices = np.argsort(scores)[::-1]
    top_indices = [i for i in top_indices if i != idx][:top_n]

    recommended = df.iloc[top_indices].copy()
    recommended["score"] = scores[top_indices]
    return recommended.reset_index(drop=True)

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("Movie Recommendation App")
st.markdown(
    "Use the selected movie as a seed and find similar films from the dataset. "
    "This app loads a precomputed similarity model from `movie_model.pkl`."
)

movies_df = load_movies(MOVIES_PATH)
similarity_matrix = load_similarity_matrix(MODEL_PATH)

if similarity_matrix.shape[0] != len(movies_df):
    st.error(
        f"The similarity matrix shape {similarity_matrix.shape} does not match the number of movies {len(movies_df)}."
    )
else:
    selected_title = st.selectbox(
        "Select a movie to get recommendations:",
        options=movies_df["title"].sort_values().unique(),
        index=0,
    )
    top_n = st.slider("Number of recommendations", min_value=5, max_value=25, value=10)

    if st.button("Recommend"):
        try:
            recommendations = get_recommendations(selected_title, movies_df, similarity_matrix, top_n)
            st.subheader("Recommended Movies")
            st.write(
                recommendations[["title", "genres", "release_date", "vote_average", "popularity", "score"]]
                .rename(columns={"vote_average": "rating", "release_date": "release"})
            )
        except Exception as e:
            st.error(f"Failed to generate recommendations: {e}")

    with st.expander("About this app"):
        st.write(
            "This application uses a precomputed cosine similarity-style matrix stored in `movie_model.pkl` "
            "to find movies that are closest to the selected movie title. The dataset is loaded from `movies.csv`."
        )
        st.write(
            "If you want to update the recommendation engine, replace `movie_model.pkl` with a new similarity matrix "
            "of the same shape and keep `movies.csv` aligned with that matrix."
        )

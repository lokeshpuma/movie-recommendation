# Movie Recommendation App

A simple Streamlit-based movie recommendation interface built on a precomputed similarity model.

## Project Files

- `app.py` - Streamlit application that loads the similarity model and movie metadata.
- `requirements.txt` - Python dependencies for the app.

> `movie_model.pkl` and `movies.csv` are large files excluded from the repository with `.gitignore`.
> Download them from the provided Google Drive links before running the app.

## Large Files

- `movie_model.pkl`: https://drive.google.com/file/d/1NT10KdgyarAsxsGCsnnfsbJ_1xb7AQtD/view?usp=drive_link
- `movies.csv`: https://drive.google.com/file/d/1Zd9dc9qh9S29pCgYDZ3wYCct0_EbsAr1/view?usp=sharing

## How to Run

1. Activate your Conda environment:
   ```bash
   conda activate tf
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## What This App Does

- Loads `movie_model.pkl` as a similarity matrix.
- Reads `movies.csv` to show movie titles and metadata.
- Lets visitors select a movie and receive similar movie recommendations.
- Displays recommended titles along with genres, release date, rating, popularity, and similarity score.

## Notes

- The app assumes `movie_model.pkl` is a square numpy array with one row/column per movie in `movies.csv`.
- If the matrix size and movie count do not match, the app reports an error.
- This interface is ready to deploy with Streamlit and can be extended with additional movie details.

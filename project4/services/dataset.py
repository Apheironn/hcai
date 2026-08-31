import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "movie_metadata.csv")

# Fixed genre order keeps the feature vector (and the learned weights) comparable across sessions.
GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Biography",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
]

NUMERIC_FEATURES = [
    ("imdb_score", "IMDb rating"),
    ("duration", "Runtime"),
    ("title_year", "Recency"),
    ("log_votes", "Popularity"),
]

# Participants can only express preferences over movies they might recognise.
MIN_VOTES = 10000
MIN_YEAR = 1970
CLIP = 3.0

_CACHE = None


class MovieDataset:
    """IMDB 5000 catalogue with a linear feature representation (Task 1)."""

    def __init__(self, frame, features, feature_names):
        self.frame = frame
        self.features = features
        self.feature_names = feature_names

    @classmethod
    def load(cls):
        global _CACHE
        if _CACHE is not None:
            return _CACHE

        if not os.path.isfile(DATA_PATH):
            raise ValueError(
                "movie_metadata.csv not found. Place the IMDB 5000 Movie Dataset in project4/data/."
            )

        raw = pd.read_csv(DATA_PATH)
        frame = cls._clean(raw)
        features, feature_names = cls._extract_features(frame)
        _CACHE = cls(frame, features, feature_names)
        return _CACHE

    @staticmethod
    def _clean(raw):
        columns = ["movie_title", "genres", "title_year", "duration", "imdb_score", "num_voted_users"]
        frame = raw.dropna(subset=columns).copy()
        frame = frame[frame["num_voted_users"] >= MIN_VOTES]
        frame = frame[frame["title_year"] >= MIN_YEAR]
        frame["movie_title"] = frame["movie_title"].str.strip()
        frame = frame.drop_duplicates(subset=["movie_title", "title_year"])
        frame["log_votes"] = np.log10(frame["num_voted_users"])
        frame["director_name"] = frame["director_name"].fillna("Unknown director")
        frame["content_rating"] = frame["content_rating"].fillna("Not rated")
        return frame.reset_index(drop=True)

    @classmethod
    def _extract_features(cls, frame):
        """Multi-hot genres + standardised numeric attributes."""
        blocks, names = [], []

        genre_lists = frame["genres"].str.split("|")
        for genre in GENRES:
            blocks.append(genre_lists.apply(lambda gs, g=genre: float(g in gs)).to_numpy())
            names.append(genre)

        for column, label in NUMERIC_FEATURES:
            values = frame[column].to_numpy(dtype=float)
            std = values.std() or 1.0
            blocks.append(np.clip((values - values.mean()) / std, -CLIP, CLIP))
            names.append(label)

        return np.column_stack(blocks), names

    @property
    def n_movies(self):
        return len(self.frame)

    @property
    def n_features(self):
        return self.features.shape[1]

    def sample(self, n, rng):
        return [int(i) for i in rng.sample(range(self.n_movies), n)]

    def card(self, index):
        """Display payload for one movie."""
        row = self.frame.iloc[index]
        genres = [g for g in str(row["genres"]).split("|")][:3]
        return {
            "index": int(index),
            "title": row["movie_title"],
            "year": int(row["title_year"]),
            "genres": ", ".join(genres),
            "director": row["director_name"],
            "duration": int(row["duration"]),
            "imdb_score": float(row["imdb_score"]),
            "content_rating": row["content_rating"],
            "link": row["movie_imdb_link"],
        }

    def cards(self, indices):
        return [self.card(i) for i in indices]

    def summary(self):
        years = self.frame["title_year"]
        return {
            "n_movies": self.n_movies,
            "n_features": self.n_features,
            "n_genres": len(GENRES),
            "year_min": int(years.min()),
            "year_max": int(years.max()),
            "min_votes": MIN_VOTES,
        }

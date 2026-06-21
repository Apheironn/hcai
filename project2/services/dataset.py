import numpy as np
import pandas as pd
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


RAW_FEATURES = [
    "island",
    "sex",
    "year",
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
NUMERIC_FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
CATEGORICAL_FEATURES = ["island", "sex", "year"]
TARGET = "species"


class PenguinDataset:
    def __init__(self, df_raw, x_train, x_test, y_train, y_test, feature_names, label_encoder, scaler):
        self.df_raw = df_raw.reset_index(drop=True)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.class_names = list(label_encoder.classes_)

        self._category_values = {
            col: sorted(df_raw[col].dropna().unique().tolist()) for col in CATEGORICAL_FEATURES
        }
        self._numeric_bounds = {
            col: (float(df_raw[col].min()), float(df_raw[col].max())) for col in NUMERIC_FEATURES
        }
        self._mad = {
            col: float(np.median(np.abs(df_raw[col] - df_raw[col].median())))
            for col in NUMERIC_FEATURES
        }
        for col in NUMERIC_FEATURES:
            if self._mad[col] == 0:
                self._mad[col] = 1.0

    @classmethod
    def load(cls):
        df = load_penguins()
        cols = [TARGET] + RAW_FEATURES
        df = df[cols].dropna().reset_index(drop=True)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[TARGET])

        encoded = pd.get_dummies(df[RAW_FEATURES], columns=CATEGORICAL_FEATURES, drop_first=False)
        feature_names = list(encoded.columns)
        x = encoded.values.astype(float)

        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.25, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        return cls(df, x_train, x_test, y_train, y_test, feature_names, label_encoder, scaler)

    @property
    def n_rows(self):
        return len(self.df_raw)

    @property
    def n_features(self):
        return len(RAW_FEATURES)

    def preview(self, rows=5):
        return self.df_raw.head(rows).to_html(classes="data-preview", index=False)

    def class_counts(self):
        return self.df_raw[TARGET].value_counts().to_dict()

    def row_raw(self, index):
        row = self.df_raw.iloc[int(index)]
        return {col: row[col] for col in RAW_FEATURES}, row[TARGET]

    def row_label(self, index):
        row = self.df_raw.iloc[int(index)]
        return (
            f"#{index} · {row[TARGET]} · {row['island']} · "
            f"{int(row['year']) if pd.notna(row['year']) else '?'}"
        )

    def encode_row(self, raw_dict):
        frame = pd.DataFrame([raw_dict])[RAW_FEATURES]
        encoded = pd.get_dummies(frame, columns=CATEGORICAL_FEATURES, drop_first=False)
        for name in self.feature_names:
            if name not in encoded.columns:
                encoded[name] = 0.0
        return encoded[self.feature_names].values[0].astype(float)

    def scale_vector(self, vector):
        return self.scaler.transform(vector.reshape(1, -1))

    def decode_vector(self, vector):
        values = {}
        vec_map = dict(zip(self.feature_names, vector))
        for col in NUMERIC_FEATURES:
            if col in vec_map:
                values[col] = round(float(vec_map[col]), 2)

        for col in CATEGORICAL_FEATURES:
            prefix = f"{col}_"
            matches = [
                name[len(prefix) :]
                for name, val in vec_map.items()
                if name.startswith(prefix) and val > 0.5
            ]
            values[col] = matches[0] if matches else self.df_raw[col].mode().iloc[0]

        return values

    def mad_weights(self):
        weights = dict(self._mad)
        for col in CATEGORICAL_FEATURES:
            weights[col] = 1.0
        return weights

    def category_values(self, col):
        return self._category_values[col]

    def numeric_bounds(self, col):
        return self._numeric_bounds[col]

    def feature_index(self, raw_feature):
        if raw_feature not in NUMERIC_FEATURES:
            raise ValueError("Select a numeric feature.")
        for idx, name in enumerate(self.feature_names):
            if name == raw_feature:
                return idx
        raise ValueError("Feature not found in encoded matrix.")

    def class_distribution_plot(self, plot_style):
        import matplotlib.pyplot as plt

        counts = self.class_counts()
        labels = [name for name in self.class_names if name in counts]
        values = [counts[name] for name in labels]
        colors = [plot_style.SPECIES_COLORS.get(name, plot_style.ACCENT) for name in labels]

        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=plot_style.DPI)
        fig.patch.set_facecolor(plot_style.BG)
        ax.set_facecolor(plot_style.BG)
        ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title("Species distribution", fontsize=13, fontweight="600", pad=12)
        ax.set_ylabel("Count")
        plot_style.apply(ax)
        fig.tight_layout()
        return plot_style.save(fig)

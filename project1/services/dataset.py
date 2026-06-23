import os
import shutil
import uuid

import pandas as pd
from django.conf import settings


ID_COLUMNS = {"id", "index"}


class Dataset:
    def __init__(self, df, target_name, problem_type):
        self.df = df
        self.target_name = target_name
        self.problem_type = problem_type
        self.features = list(df.columns[:-1])

    @property
    def target(self):
        return self.df.iloc[:, -1]

    @property
    def n_rows(self):
        return len(self.df)

    @classmethod
    def from_path(cls, path, problem_type="auto"):
        if not os.path.exists(path):
            raise ValueError("Dataset file not found.")

        df = pd.read_csv(path)
        if df.empty or len(df.columns) < 2:
            raise ValueError("CSV must have a header row and at least one data row.")

        df = cls._drop_id_columns(df)
        target_name = df.columns[-1]
        features = list(df.columns[:-1])

        if not features:
            raise ValueError("CSV must have at least one feature column.")

        for col in features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError("All feature columns must be numeric.")

        resolved_type = cls._resolve_problem_type(df[target_name], problem_type)
        return cls(df, target_name, resolved_type)

    @classmethod
    def from_upload(cls, uploaded_file, problem_type="auto"):
        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)

        path = os.path.join(upload_dir, f"{uuid.uuid4()}.csv")
        with open(path, "wb") as handle:
            for chunk in uploaded_file.chunks():
                handle.write(chunk)

        return cls.from_path(path, problem_type), path

    @classmethod
    def from_sample(cls, problem_type="auto"):
        sample_path = os.path.join(
            settings.BASE_DIR, "project1", "static", "project1", "sample_iris.csv"
        )
        upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        path = os.path.join(upload_dir, f"sample_iris_{uuid.uuid4().hex}.csv")
        shutil.copy(sample_path, path)
        return cls.from_path(path, problem_type), path

    @staticmethod
    def _drop_id_columns(df):
        cols = [c for c in df.columns if c.lower() not in ID_COLUMNS]
        return df[cols]

    @staticmethod
    def _resolve_problem_type(target, override):
        if override in ("classification", "regression"):
            return override

        if not pd.api.types.is_numeric_dtype(target):
            return "classification"

        if target.nunique() <= 20:
            return "classification"

        return "regression"

    def preview(self, rows=5):
        return self.df.head(rows).to_html(classes="data-preview", index=False)

    def to_session(self, path):
        return {
            "path": path,
            "features": self.features,
            "target": self.target_name,
            "problem_type": self.problem_type,
            "n_rows": self.n_rows,
        }

    @classmethod
    def from_session(cls, session_data, problem_type="auto"):
        if not session_data or "path" not in session_data:
            raise ValueError("Upload a CSV first.")
        override = problem_type if problem_type != "auto" else session_data.get("problem_type", "auto")
        return cls.from_path(session_data["path"], override)

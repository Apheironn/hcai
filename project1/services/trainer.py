from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


CLASSIFICATION_MODELS = {
    "LogisticRegression": {
        "class": LogisticRegression,
        "param": "C",
        "defaults": "0.01,0.1,1",
        "kwargs": {"max_iter": 1000},
    },
    "KNeighborsClassifier": {
        "class": KNeighborsClassifier,
        "param": "n_neighbors",
        "defaults": "3,5,7,15",
        "kwargs": {},
    },
    "DecisionTreeClassifier": {
        "class": DecisionTreeClassifier,
        "param": "max_depth",
        "defaults": "2,5,10,20",
        "kwargs": {},
    },
}

REGRESSION_MODELS = {
    "LinearRegression": {
        "class": LinearRegression,
        "param": None,
        "defaults": "",
        "kwargs": {},
    },
    "Ridge": {
        "class": Ridge,
        "param": "alpha",
        "defaults": "0.1,1,10",
        "kwargs": {},
    },
    "KNeighborsRegressor": {
        "class": KNeighborsRegressor,
        "param": "n_neighbors",
        "defaults": "3,5,7,15",
        "kwargs": {},
    },
}

CLASSIFICATION_METRICS = {
    "accuracy": accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
}

REGRESSION_METRICS = {
    "r2": r2_score,
    "mse": mean_squared_error,
}

OVERFIT_GAP = 0.08


class ModelTrainer:
    @classmethod
    def models_for(cls, problem_type):
        if problem_type == "classification":
            return CLASSIFICATION_MODELS
        return REGRESSION_MODELS

    @classmethod
    def metrics_for(cls, problem_type):
        if problem_type == "classification":
            return list(CLASSIFICATION_METRICS.keys())
        return list(REGRESSION_METRICS.keys())

    @classmethod
    def model_config(cls, problem_type):
        registry = cls.models_for(problem_type)
        return {
            name: {
                "param": cfg["param"] or "",
                "defaults": cfg["defaults"],
            }
            for name, cfg in registry.items()
        }

    @classmethod
    def param_name_for(cls, model_name, problem_type):
        registry = cls.models_for(problem_type)
        if model_name not in registry:
            raise ValueError("Select a valid model.")
        return registry[model_name]["param"]

    @classmethod
    def default_values_for(cls, model_name, problem_type):
        registry = cls.models_for(problem_type)
        if model_name not in registry:
            return ""
        return registry[model_name]["defaults"]

    @classmethod
    def run(cls, dataset, model_name, test_size, param_values, metric):
        registry = cls.models_for(dataset.problem_type)
        if model_name not in registry:
            raise ValueError("Select a valid model.")

        metrics = (
            CLASSIFICATION_METRICS
            if dataset.problem_type == "classification"
            else REGRESSION_METRICS
        )
        if metric not in metrics:
            raise ValueError("Select a valid metric.")

        cfg = registry[model_name]
        param_name = cfg["param"]
        score_fn = metrics[metric]
        lower_is_better = metric == "mse"

        x = dataset.df[dataset.features].values
        y = dataset.target.values
        if dataset.problem_type == "classification" and y.dtype == object:
            y = LabelEncoder().fit_transform(y)

        split_kwargs = {
            "test_size": (100 - test_size) / 100,
            "random_state": 42,
        }
        if dataset.problem_type == "classification" and len(set(y)) > 1:
            split_kwargs["stratify"] = y

        x_train, x_test, y_train, y_test = train_test_split(x, y, **split_kwargs)

        values = cls._parse_param_values(param_values, param_name)
        results = []

        for value in values:
            kwargs = dict(cfg["kwargs"])
            if param_name:
                kwargs[param_name] = value
            model = cfg["class"](**kwargs)
            model.fit(x_train, y_train)
            train_score = float(score_fn(y_train, model.predict(x_train)))
            test_score = float(score_fn(y_test, model.predict(x_test)))
            gap = round(train_score - test_score, 4)
            if lower_is_better:
                gap = round(test_score - train_score, 4)
            results.append(
                {
                    "param": value if param_name else "default",
                    "train_score": round(train_score, 4),
                    "test_score": round(test_score, 4),
                    "gap": gap,
                    "overfit": gap > OVERFIT_GAP,
                }
            )

        best = cls._pick_best(results, lower_is_better)
        for row in results:
            row["best"] = row is best

        return {
            "results": results,
            "param_name": param_name or "param",
            "metric": metric,
            "model": model_name,
            "lower_is_better": lower_is_better,
            "train_rows": len(y_train),
            "test_rows": len(y_test),
            "overfit_warning": any(row["overfit"] for row in results),
        }

    @staticmethod
    def _parse_param_values(raw, param_name):
        if not param_name:
            return [None]
        if not raw or not raw.strip():
            raise ValueError("Enter at least one hyperparameter value.")
        values = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if part.lower() == "none":
                values.append(None)
                continue
            try:
                values.append(int(part) if "." not in part else float(part))
            except ValueError as exc:
                raise ValueError("Invalid value in hyperparameter list.") from exc
        if not values:
            raise ValueError("Enter at least one hyperparameter value.")
        return values

    @staticmethod
    def _pick_best(results, lower_is_better):
        key = "test_score"
        if lower_is_better:
            return min(results, key=lambda row: row[key])
        return max(results, key=lambda row: row[key])

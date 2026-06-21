from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


CLASSIFICATION_MODELS = {
    "LogisticRegression": (LogisticRegression, "C", max),
    "KNeighborsClassifier": (KNeighborsClassifier, "n_neighbors", max),
    "DecisionTreeClassifier": (DecisionTreeClassifier, "max_depth", max),
}

REGRESSION_MODELS = {
    "LinearRegression": (LinearRegression, None, max),
    "Ridge": (Ridge, "alpha", max),
    "KNeighborsRegressor": (KNeighborsRegressor, "n_neighbors", max),
}

CLASSIFICATION_METRICS = {
    "accuracy": accuracy_score,
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
}

REGRESSION_METRICS = {
    "r2": r2_score,
    "mse": mean_squared_error,
}


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
    def param_name_for(cls, model_name, problem_type):
        registry = cls.models_for(problem_type)
        if model_name not in registry:
            raise ValueError("Select a valid model.")
        return registry[model_name][1]

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

        model_cls, param_name, best_fn = registry[model_name]
        score_fn = metrics[metric]
        lower_is_better = metric == "mse"

        x = dataset.df[dataset.features].values
        y = dataset.target.values
        if dataset.problem_type == "classification" and y.dtype == object:
            y = LabelEncoder().fit_transform(y)

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=(100 - test_size) / 100,
            random_state=42,
        )

        values = cls._parse_param_values(param_values, param_name)
        results = []

        for value in values:
            kwargs = {param_name: value} if param_name else {}
            model = model_cls(**kwargs)
            model.fit(x_train, y_train)
            train_score = score_fn(y_train, model.predict(x_train))
            test_score = score_fn(y_test, model.predict(x_test))
            results.append(
                {
                    "param": value if param_name else "default",
                    "train_score": round(float(train_score), 4),
                    "test_score": round(float(test_score), 4),
                }
            )

        best = cls._pick_best(results, lower_is_better)
        for row in results:
            row["best"] = row is best

        return {
            "results": results,
            "param_name": param_name or "param",
            "metric": metric,
            "lower_is_better": lower_is_better,
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

class ModelSelector:
    @staticmethod
    def score(candidate, lambda_):
        return candidate.test_accuracy - lambda_ * candidate.complexity

    @classmethod
    def select(cls, candidates, lambda_):
        if not candidates:
            raise ValueError("No models available.")
        return max(
            candidates,
            key=lambda c: (cls.score(c, lambda_), c.test_accuracy, -c.complexity),
        )

    @classmethod
    def scored_candidates(cls, candidates, lambda_):
        rows = []
        for candidate in candidates:
            rows.append(
                {
                    "id": candidate.id,
                    "param_label": candidate.param_label,
                    "param_value": candidate.param_value,
                    "test_accuracy": candidate.test_accuracy,
                    "complexity": candidate.complexity,
                    "score": round(cls.score(candidate, lambda_), 4),
                }
            )
        selected = cls.select(candidates, lambda_)
        for row in rows:
            row["selected"] = row["id"] == selected.id
        return rows, selected

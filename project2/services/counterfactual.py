import copy

import numpy as np


class CounterfactualFinder:
    DEFAULT_N = 2000
    DEFAULT_K = 5
    DEFAULT_SIGMA = 0.3

    @classmethod
    def find(cls, model, dataset, row_index, target_class, k=None, n=None, sigma=None):
        k = k or cls.DEFAULT_K
        n = n or cls.DEFAULT_N
        sigma = sigma or cls.DEFAULT_SIGMA

        raw, current_label = dataset.row_raw(row_index)
        if target_class not in dataset.class_names:
            raise ValueError("Select a valid target species.")
        if target_class == current_label:
            raise ValueError("Target species must differ from the current label.")

        target_id = dataset.label_encoder.transform([target_class])[0]
        original = dataset.encode_row(raw)
        found = []

        for attempt in range(4):
            batch = cls._sample_batch(model, dataset, raw, original, target_id, n, sigma)
            found.extend(batch)
            if found:
                break
            n *= 2
            sigma += 0.15

        if not found:
            return []

        found.sort(key=lambda item: item["distance"])
        seen = set()
        unique = []
        for item in found:
            key = tuple(sorted(item["values"].items()))
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
            if len(unique) >= k:
                break
        return unique

    @classmethod
    def _sample_batch(cls, model, dataset, raw, original, target_id, n, sigma):
        hits = []
        rng = np.random.default_rng(42)
        mad = dataset.mad_weights()

        for _ in range(n):
            candidate_raw = copy.deepcopy(raw)
            for col in NUMERIC_FEATURES:
                low, high = dataset.numeric_bounds(col)
                noise = rng.normal(0, sigma * mad[col])
                candidate_raw[col] = float(np.clip(raw[col] + noise, low, high))

            for col in ["island", "sex", "year"]:
                if rng.random() < 0.5:
                    options = [v for v in dataset.category_values(col) if v != raw[col]]
                    if options:
                        candidate_raw[col] = rng.choice(options)

            vector = dataset.encode_row(candidate_raw)
            scaled = dataset.scale_vector(vector)
            pred = int(model.predict(scaled)[0])
            if pred != target_id:
                continue

            distance = cls._distance(raw, candidate_raw, mad)
            hits.append(
                {
                    "values": {key: candidate_raw[key] for key in raw},
                    "distance": round(distance, 4),
                    "predicted": dataset.label_encoder.inverse_transform([pred])[0],
                }
            )
        return hits

    @staticmethod
    def _distance(original, candidate, mad):
        total = 0.0
        for col, weight in mad.items():
            if col in ["island", "sex", "year"]:
                if original[col] != candidate[col]:
                    total += 1.0 / weight
            else:
                total += abs(float(original[col]) - float(candidate[col])) / weight
        return total

    @classmethod
    def build_diff_rows(cls, original_raw, counterfactual):
        rows = []
        for col in original_raw:
            orig = original_raw[col]
            new = counterfactual["values"][col]
            changed = orig != new
            delta = ""
            if col not in ["island", "sex", "year"] and changed:
                delta = f"{float(new) - float(orig):+.1f}"
            rows.append(
                {
                    "feature": col,
                    "original": orig,
                    "counterfactual": new,
                    "delta": delta,
                    "changed": changed,
                }
            )
        return rows

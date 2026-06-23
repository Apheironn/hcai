import numpy as np

from .dataset import CLASS_NAMES


BATCH_SIZE = 5


class HumanExpertSession:
    @classmethod
    def start_batch(cls, dataset, store, seed=42):
        rng = np.random.default_rng(seed)
        labeled = set(int(k) for k in store.data.get("human_labels", {}).keys())
        pool = [i for i in range(dataset.n_train) if i not in labeled]
        if len(pool) < BATCH_SIZE:
            pool = list(range(min(BATCH_SIZE, dataset.n_train)))
        chosen = rng.choice(pool, size=min(BATCH_SIZE, len(pool)), replace=False)
        indices = [int(i) for i in chosen]
        store.data["human_batch_indices"] = indices
        return indices

    @classmethod
    def record_label(cls, store, index, class_name):
        if class_name not in CLASS_NAMES:
            raise ValueError("Select a valid class.")
        labels = store.data.setdefault("human_labels", {})
        labels[str(int(index))] = CLASS_NAMES.index(class_name)
        store.data["human_labels"] = labels

    @classmethod
    def batch_progress(cls, store):
        indices = store.data.get("human_batch_indices", [])
        labels = store.data.get("human_labels", {})
        done = sum(1 for i in indices if str(i) in labels)
        return done, len(indices)

    @classmethod
    def build_report(cls, dataset, store, expert):
        labels = store.data.get("human_labels", {})
        if not labels:
            raise ValueError("Label at least one article first.")

        indices = [int(k) for k in labels.keys()]
        texts = [dataset.train_texts[i] for i in indices]
        true_y = dataset.train_labels[indices]
        human_preds = np.array([labels[str(i)] for i in indices])
        expert_preds = expert.predict_batch(texts)

        human_acc = float(np.mean(human_preds == true_y))
        expert_acc = float(np.mean(expert_preds == true_y))
        per_class = {}
        for idx, name in enumerate(CLASS_NAMES):
            mask = true_y == idx
            if mask.sum() == 0:
                per_class[name] = {"human": None, "expert": None}
            else:
                per_class[name] = {
                    "human": round(float(np.mean(human_preds[mask] == true_y[mask])), 4),
                    "expert": round(float(np.mean(expert_preds[mask] == true_y[mask])), 4),
                }

        report = {
            "n_labeled": len(indices),
            "human_accuracy": round(human_acc, 4),
            "expert_accuracy": round(expert_acc, 4),
            "per_class": per_class,
        }
        store.data["human_report"] = report
        return report

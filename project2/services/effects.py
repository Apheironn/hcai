import matplotlib.pyplot as plt
import numpy as np

from .plot_style import PlotStyle


class EffectPlots:
    GRID_SIZE = 50
    TREE_BINS = 20

    @classmethod
    def render(cls, model, dataset, candidate, raw_feature, model_type, lambda_):
        feature_idx = dataset.feature_index(raw_feature)
        grid = cls._grid(dataset, feature_idx)
        pdp = cls._pdp(model, dataset.x_train, feature_idx, grid)
        if model_type == "logreg":
            ale = cls._ale_logreg(model, dataset.x_train, feature_idx)
            ale_note = "analytic gradient (logreg)"
        else:
            ale = cls._ale_tree(model, dataset.x_train, feature_idx)
            ale_note = "finite differences over quantile bins (tree)"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=PlotStyle.DPI)
        fig.patch.set_facecolor(PlotStyle.BG)

        for ax in axes:
            ax.set_facecolor(PlotStyle.BG)

        for class_idx, class_name in enumerate(dataset.class_names):
            color = PlotStyle.SPECIES_COLORS.get(class_name, PlotStyle.ACCENT)
            axes[0].plot(grid, pdp[:, class_idx], label=class_name, color=color, linewidth=2)
            axes[1].plot(ale["grid"], ale["values"][:, class_idx], label=class_name, color=color, linewidth=2)

        axes[0].set_title("PDP", fontsize=12, fontweight="600")
        axes[1].set_title(f"ALE ({ale_note})", fontsize=12, fontweight="600")
        axes[0].set_xlabel(raw_feature)
        axes[1].set_xlabel(raw_feature)
        axes[0].set_ylabel("P(species)")
        axes[1].set_ylabel("Accumulated effect")
        axes[0].set_ylim(0, 1)
        for ax in axes:
            PlotStyle.apply(ax)
        axes[1].legend(title="Species", loc="best", fontsize=8)
        fig.suptitle(
            f"{model_type}, lambda={lambda_}, feature={raw_feature}, "
            f"Omega={candidate.complexity}, acc={candidate.test_accuracy}",
            fontsize=11,
            y=1.02,
        )
        fig.tight_layout()
        return PlotStyle.save(fig), ale_note

    @classmethod
    def _grid(cls, dataset, feature_idx):
        values = dataset.x_train[:, feature_idx]
        low, high = np.quantile(values, [0.05, 0.95])
        if low == high:
            high = low + 1.0
        return np.linspace(low, high, cls.GRID_SIZE)

    @classmethod
    def _pdp(cls, model, x_train, feature_idx, grid):
        curves = []
        for value in grid:
            x_copy = x_train.copy()
            x_copy[:, feature_idx] = value
            curves.append(model.predict_proba(x_copy).mean(axis=0))
        return np.array(curves)

    @classmethod
    def _ale_logreg(cls, model, x_train, feature_idx):
        """ALE from the analytic partial derivative of the softmax.

        For p_k = softmax(W x + b) we have, in closed form,
            d p_k / d x_j = p_k * (w_kj - sum_c p_c w_cj),
        so no finite differences are needed: we just integrate this derivative
        over quantile bins of the feature.
        """
        values = x_train[:, feature_idx]
        bin_edges = np.unique(np.quantile(values, np.linspace(0, 1, cls.TREE_BINS + 1)))
        n_classes = len(model.classes_)

        coef = model.coef_
        if coef.shape[0] == 1:  # binary sigmoid -> equivalent two-row softmax [0, w]
            coef = np.vstack([np.zeros_like(coef), coef])
        w_j = coef[:, feature_idx]                       # (n_classes,)

        proba = model.predict_proba(x_train)            # (n, n_classes)
        mean_w = proba @ w_j                            # (n,) = sum_c p_c w_cj
        grad = proba * (w_j[None, :] - mean_w[:, None])  # (n, n_classes) = d p / d x_j

        ale = np.zeros((len(bin_edges), n_classes))
        for i in range(1, len(bin_edges)):
            lower, upper = bin_edges[i - 1], bin_edges[i]
            mask = (values >= lower) & (values <= upper)
            step = grad[mask].mean(axis=0) * (upper - lower) if np.any(mask) else 0.0
            ale[i] = ale[i - 1] + step

        ale -= ale.mean(axis=0)  # centre the accumulated effect
        return {"grid": bin_edges, "values": ale}

    @classmethod
    def _ale_tree(cls, model, x_train, feature_idx):
        values = x_train[:, feature_idx]
        order = np.argsort(values)
        sorted_vals = values[order]
        sorted_x = x_train[order].copy()
        bin_edges = np.unique(np.quantile(sorted_vals, np.linspace(0, 1, cls.TREE_BINS + 1)))
        ale = np.zeros((len(bin_edges), len(model.classes_)))

        for class_idx in range(len(model.classes_)):
            accumulated = 0.0
            ale[0, class_idx] = 0.0
            for i in range(1, len(bin_edges)):
                lower, upper = bin_edges[i - 1], bin_edges[i]
                mask = (sorted_vals >= lower) & (sorted_vals <= upper)
                if not np.any(mask):
                    ale[i, class_idx] = accumulated
                    continue
                local_x = sorted_x[mask].copy()
                lower_x = local_x.copy()
                upper_x = local_x.copy()
                lower_x[:, feature_idx] = lower
                upper_x[:, feature_idx] = upper
                p_lower = cls._class_prob(model, lower_x, class_idx)
                p_upper = cls._class_prob(model, upper_x, class_idx)
                accumulated += np.mean(p_upper - p_lower)
                ale[i, class_idx] = accumulated

        ale -= ale.mean(axis=0)  # centre the accumulated effect
        return {"grid": bin_edges, "values": ale}

    @staticmethod
    def _class_prob(model, x_batch, class_idx):
        probs = model.predict_proba(x_batch)
        return probs[:, class_idx]

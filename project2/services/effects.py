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
            ale = cls._ale_logreg(model, dataset.x_train, feature_idx, grid)
            ale_note = "exact gradients (logreg)"
        else:
            ale = cls._ale_tree(model, dataset.x_train, feature_idx)
            ale_note = "quantile bins (tree)"

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=PlotStyle.DPI)
        fig.patch.set_facecolor(PlotStyle.BG)

        for ax in axes:
            ax.set_facecolor(PlotStyle.BG)

        for class_idx, class_name in enumerate(dataset.class_names):
            color = PlotStyle.SPECIES_COLORS.get(class_name, PlotStyle.ACCENT)
            axes[0].plot(grid, pdp[:, class_idx], label=class_name, color=color, linewidth=2)
            x_ale = grid if model_type == "logreg" else ale["grid"]
            y_ale = ale["values"][:, class_idx]
            axes[1].plot(x_ale, y_ale, label=class_name, color=color, linewidth=2)

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
    def _ale_logreg(cls, model, x_train, feature_idx, grid):
        coef = model.coef_[:, feature_idx]
        intercept = model.intercept_
        values = x_train[:, feature_idx]
        order = np.argsort(values)
        sorted_vals = values[order]
        sorted_x = x_train[order].copy()

        n_bins = min(cls.TREE_BINS, len(sorted_vals) - 1)
        bin_edges = np.quantile(sorted_vals, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
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
                local_lower = local_x.copy()
                local_upper = local_x.copy()
                local_lower[:, feature_idx] = lower
                local_upper[:, feature_idx] = upper
                p_lower = cls._class_prob(model, local_lower, class_idx)
                p_upper = cls._class_prob(model, local_upper, class_idx)
                local_effect = np.mean(p_upper - p_lower)
                accumulated += local_effect
                ale[i, class_idx] = accumulated

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

        return {"grid": bin_edges, "values": ale}

    @staticmethod
    def _class_prob(model, x_batch, class_idx):
        probs = model.predict_proba(x_batch)
        return probs[:, class_idx]

import os
import uuid

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from django.conf import settings
from project2.services.plot_style import PlotStyle
from sklearn.metrics import confusion_matrix


CLASS_COLORS = list(mcolors.TABLEAU_COLORS.values())


class PlotBuilder:
    @classmethod
    def build(cls, dataset, x_feature, y_feature=None):
        if x_feature not in dataset.features:
            raise ValueError("Select a valid X feature.")

        plot_dir = os.path.join(settings.MEDIA_ROOT, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join(plot_dir, filename)

        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=140)
        fig.patch.set_facecolor(PlotStyle.BG)
        ax.set_facecolor(PlotStyle.BG)
        target = dataset.target

        if dataset.problem_type == "regression" and not y_feature:
            ax.scatter(
                dataset.df[x_feature],
                target,
                alpha=0.75,
                s=48,
                c=PlotStyle.ACCENT,
                edgecolors="white",
                linewidths=0.6,
            )
            ax.set_xlabel(x_feature, fontsize=11, labelpad=8)
            ax.set_ylabel(dataset.target_name, fontsize=11, labelpad=8)
            ax.set_title(
                f"{x_feature} vs {dataset.target_name}",
                fontsize=13,
                fontweight="600",
                pad=12,
            )
        else:
            if not y_feature or y_feature not in dataset.features:
                raise ValueError("Select two features for this plot.")
            labels = target.astype(str)
            for index, label in enumerate(sorted(labels.unique())):
                mask = labels == label
                ax.scatter(
                    dataset.df.loc[mask, x_feature],
                    dataset.df.loc[mask, y_feature],
                    label=label,
                    alpha=0.82,
                    s=52,
                    c=CLASS_COLORS[index % len(CLASS_COLORS)],
                    edgecolors="white",
                    linewidths=0.7,
                )
            ax.set_xlabel(x_feature, fontsize=11, labelpad=8)
            ax.set_ylabel(y_feature, fontsize=11, labelpad=8)
            ax.set_title(
                f"{x_feature} vs {y_feature}",
                fontsize=13,
                fontweight="600",
                pad=12,
            )
            ax.legend(
                title=dataset.target_name,
                frameon=True,
                fancybox=True,
                shadow=False,
                borderpad=0.8,
                loc="best",
            )

        PlotStyle.apply(ax)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        return settings.MEDIA_URL + f"plots/{filename}"

    @classmethod
    def class_distribution(cls, dataset):
        if dataset.problem_type != "classification":
            return None
        counts = dataset.target.astype(str).value_counts()
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=PlotStyle.BG)
        names = list(counts.index)
        values = list(counts.values)
        bars = ax.bar(names, values, color=CLASS_COLORS[: len(names)], alpha=0.9, edgecolor="white")
        ax.set_title(f"Target: {dataset.target_name}", fontsize=12, weight="bold", color="#1a4480")
        ax.set_ylabel("Count")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=8)
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def histogram(cls, dataset, x_feature):
        if x_feature not in dataset.features:
            raise ValueError("Select a valid feature.")
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=PlotStyle.BG)
        ax.hist(dataset.df[x_feature], bins=min(20, dataset.n_rows), color=PlotStyle.ACCENT, alpha=0.85, edgecolor="white")
        ax.set_xlabel(x_feature)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Distribution of {x_feature}", fontsize=12, weight="bold", color="#1a4480")
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def param_sweep(cls, results, param_name, metric, lower_is_better=False):
        params = [str(r["param"]) for r in results]
        test_scores = [r["test_score"] for r in results]
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=PlotStyle.BG)
        ax.plot(params, test_scores, marker="o", color=PlotStyle.ACCENT, linewidth=2)
        best_idx = int(np.argmin(test_scores) if lower_is_better else np.argmax(test_scores))
        ax.scatter([params[best_idx]], [test_scores[best_idx]], s=120, c="#59A14F", zorder=5, label="Best")
        ax.set_xlabel(param_name)
        ax.set_ylabel(f"Test {metric}")
        ax.set_title("Hyperparameter sweep", fontsize=12, weight="bold", color="#1a4480")
        ax.legend(frameon=False)
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def confusion_matrix_plot(cls, y_true, y_pred, labels):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=PlotStyle.BG)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels([str(l) for l in labels], fontsize=8)
        ax.set_yticklabels([str(l) for l in labels], fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion matrix (best model)", fontsize=12, weight="bold", color="#1a4480")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def residuals(cls, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        residuals = y_true - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=PlotStyle.BG)
        axes[0].scatter(y_pred, y_true, alpha=0.7, c=PlotStyle.ACCENT, edgecolors="white")
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        axes[0].plot(lims, lims, "--", color="#94a3b8")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_title("Predicted vs actual")
        axes[1].scatter(y_pred, residuals, alpha=0.7, c=PlotStyle.NEGATIVE, edgecolors="white")
        axes[1].axhline(0, color="#94a3b8", linestyle="--")
        axes[1].set_xlabel("Predicted")
        axes[1].set_ylabel("Residual")
        axes[1].set_title("Residual plot")
        for ax in axes:
            PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

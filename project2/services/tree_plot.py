import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from .plot_style import PlotStyle


class TreePlotter:
    @classmethod
    def render(cls, model, dataset, candidate):
        fig, ax = plt.subplots(figsize=(14, 8), dpi=PlotStyle.DPI)
        fig.patch.set_facecolor(PlotStyle.BG)
        ax.set_facecolor(PlotStyle.BG)
        plot_tree(
            model,
            feature_names=dataset.feature_names,
            class_names=dataset.class_names,
            filled=True,
            rounded=True,
            fontsize=9,
            ax=ax,
        )
        ax.set_title(
            f"Decision Tree ({candidate.param_label}={candidate.param_value}) · "
            f"acc={candidate.test_accuracy} · leaves={int(candidate.complexity)}",
            fontsize=12,
            fontweight="600",
            pad=12,
        )
        fig.tight_layout()
        return PlotStyle.save(fig)

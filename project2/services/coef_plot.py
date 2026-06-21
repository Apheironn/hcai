import matplotlib.pyplot as plt
import numpy as np

from .plot_style import PlotStyle


class CoefficientPlotter:
    @classmethod
    def render(cls, model, dataset, candidate):
        coefs = model.coef_
        if coefs.ndim > 1:
            importance = np.mean(np.abs(coefs), axis=0)
            signed = np.mean(coefs, axis=0)
        else:
            importance = np.abs(coefs)
            signed = coefs

        order = np.argsort(importance)[::-1][:15]
        names = [dataset.feature_names[i] for i in order]
        values = [signed[i] for i in order]
        colors = [PlotStyle.POSITIVE if v >= 0 else PlotStyle.NEGATIVE for v in values]

        fig, ax = plt.subplots(figsize=(10, 6), dpi=PlotStyle.DPI)
        fig.patch.set_facecolor(PlotStyle.BG)
        ax.set_facecolor(PlotStyle.BG)
        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="#94a3b8", linewidth=0.8)
        ax.set_xlabel("Coefficient (mean across classes)")
        ax.set_title(
            f"Logistic Regression (C={candidate.param_value}) · "
            f"acc={candidate.test_accuracy} · L1={candidate.complexity}",
            fontsize=12,
            fontweight="600",
            pad=12,
        )
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

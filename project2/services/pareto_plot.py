import os

import matplotlib.pyplot as plt
import numpy as np

from .plot_style import PlotStyle
from .selector import ModelSelector


class ParetoPlotter:
    @classmethod
    def render(cls, candidates, lambda_value, selected_id):
        if not candidates:
            raise ValueError("No candidates to plot.")

        xs = [c.complexity for c in candidates]
        ys = [c.test_accuracy for c in candidates]
        scores = [ModelSelector.score(c, lambda_value) for c in candidates]

        fig, ax = plt.subplots(figsize=(7, 5), dpi=PlotStyle.DPI)
        fig.patch.set_facecolor(PlotStyle.BG)
        ax.set_facecolor(PlotStyle.BG)
        ax.scatter(xs, ys, c=PlotStyle.ACCENT, alpha=0.55, s=60, edgecolors="white")

        best_idx = int(np.argmax(scores))
        ax.scatter(
            [xs[best_idx]],
            [ys[best_idx]],
            c="#59A14F",
            s=160,
            edgecolors="white",
            linewidths=1.5,
            label="λ-selected",
            zorder=5,
        )

        for i, c in enumerate(candidates):
            if c.id == selected_id and i != best_idx:
                ax.scatter([xs[i]], [ys[i]], c="#F28E2B", s=120, zorder=4)

        ax.set_xlabel("Ω (complexity)")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"Accuracy–complexity tradeoff (λ={lambda_value})", fontsize=12, weight="bold")
        ax.legend(frameon=False, fontsize=9)
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

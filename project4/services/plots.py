import numpy as np
import matplotlib.pyplot as plt

from project2.services.plot_style import PlotStyle

DESIGN_COLORS = {"pairwise": PlotStyle.ACCENT, "ranking": "#59A14F"}


class StudyPlots:
    @classmethod
    def learned_weights(cls, results):
        """Estimated preference vector per elicitation design."""
        names = results["feature_names"]
        designs = results["designs"]
        x = np.arange(len(names))
        width = 0.38

        fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=PlotStyle.BG)
        for offset, design in zip((-width / 2, width / 2), results["order"]):
            ax.bar(
                x + offset,
                designs[design]["weights"],
                width,
                label=designs[design]["label"],
                color=DESIGN_COLORS.get(design, PlotStyle.ACCENT),
                alpha=0.9,
            )
        ax.axhline(0, color="#94a3b8", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Weight")
        ax.set_title("Estimated preference vector w", fontsize=12, weight="bold", color="#1a4480")
        ax.legend(frameon=False, fontsize=8)
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def design_comparison(cls, results):
        """Outcome measures: holdout accuracy and elicitation time."""
        designs = results["designs"]
        order = results["order"]
        labels = [designs[d]["label"].split("—")[0].strip() for d in order]
        colors = [DESIGN_COLORS.get(d, PlotStyle.ACCENT) for d in order]

        fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor=PlotStyle.BG)

        accuracies = [designs[d]["holdout_accuracy"] or 0.0 for d in order]
        bars = axes[0].bar(labels, accuracies, color=colors, alpha=0.9, edgecolor="white")
        axes[0].axhline(0.5, color="#94a3b8", linestyle="--", linewidth=1.2, label="Chance")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("Held-out pair accuracy")
        axes[0].set_title("Predictive quality", fontsize=11, weight="bold", color="#1a4480")
        axes[0].legend(frameon=False, fontsize=8)
        cls._label(axes[0], bars, "{:.2f}")

        seconds = [designs[d]["seconds"] for d in order]
        bars2 = axes[1].bar(labels, seconds, color=colors, alpha=0.9, edgecolor="white")
        axes[1].set_ylabel("Seconds")
        axes[1].set_title("Time on task", fontsize=11, weight="bold", color="#1a4480")
        cls._label(axes[1], bars2, "{:.0f}")

        for ax in axes:
            PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @staticmethod
    def _label(ax, bars, fmt):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                fmt.format(bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=8,
                color="#334155",
            )

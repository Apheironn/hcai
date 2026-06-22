import matplotlib.pyplot as plt

from project2.services.plot_style import PlotStyle

from .dataset import CLASS_NAMES


class ResultPlots:
    @classmethod
    def class_distribution(cls, counts):
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PlotStyle.BG)
        names = list(counts.keys())
        values = list(counts.values())
        ax.bar(names, values, color=PlotStyle.ACCENT, alpha=0.85)
        ax.set_title("Train class distribution")
        ax.set_ylabel("Count")
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def expert_per_class(cls, per_class_accuracy):
        fig, ax = plt.subplots(figsize=(6, 3.5), facecolor=PlotStyle.BG)
        names = list(per_class_accuracy.keys())
        values = list(per_class_accuracy.values())
        ax.bar(names, values, color=PlotStyle.ACCENT, alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_title("Expert accuracy by class (test)")
        ax.set_ylabel("Accuracy")
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def active_curves(cls, entropy_steps, random_steps):
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=PlotStyle.BG)
        ax.plot(
            [s.n_queries for s in entropy_steps],
            [s.team_accuracy for s in entropy_steps],
            marker="o",
            markersize=3,
            label="Entropy + rejector uncertainty",
            color=PlotStyle.ACCENT,
        )
        ax.plot(
            [s.n_queries for s in random_steps],
            [s.team_accuracy for s in random_steps],
            marker="s",
            markersize=3,
            label="Random baseline",
            color=PlotStyle.NEGATIVE,
        )
        ax.set_xlabel("Expert queries")
        ax.set_ylabel("Team accuracy (test)")
        ax.set_title("Active learning curves")
        ax.legend(fontsize=8)
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

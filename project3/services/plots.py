import numpy as np
import matplotlib.pyplot as plt

from project2.services.plot_style import PlotStyle

from .dataset import CLASS_NAMES


CLASS_COLORS = {
    "World": "#4E79A7",
    "Sports": "#59A14F",
    "Business": "#F28E2B",
    "Sci/Tech": "#E15759",
}


def _colors(names):
    return [CLASS_COLORS.get(name, PlotStyle.ACCENT) for name in names]


def _label_bars(ax, bars, fmt="{:.0f}"):
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )


class ResultPlots:
    @classmethod
    def class_distribution(cls, counts):
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=PlotStyle.BG)
        names = list(counts.keys())
        values = list(counts.values())
        bars = ax.bar(names, values, color=_colors(names), alpha=0.9, edgecolor="white", linewidth=0.8)
        ax.set_title("Train class distribution", fontsize=12, weight="bold", color="#1a4480")
        ax.set_ylabel("Articles")
        _label_bars(ax, bars)
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def expert_vs_classifier(cls, expert_acc, classifier_acc):
        fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=PlotStyle.BG)
        names = list(expert_acc.keys())
        x = np.arange(len(names))
        width = 0.34
        bars1 = ax.bar(x - width / 2, [expert_acc[n] for n in names], width, label="Expert", color="#59A14F", alpha=0.9)
        bars2 = ax.bar(x + width / 2, [classifier_acc[n] for n in names], width, label="Classifier", color=PlotStyle.ACCENT, alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.05)
        ax.set_title("Per-class accuracy on test set", fontsize=12, weight="bold", color="#1a4480")
        ax.set_ylabel("Accuracy")
        ax.legend(frameon=False, fontsize=9)
        for bars in (bars1, bars2):
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="#334155",
                )
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def defer_summary(cls, metrics):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), facecolor=PlotStyle.BG)

        acc_names = ["Team", "Classifier", "Expert"]
        acc_values = [
            metrics["team_accuracy"],
            metrics["classifier_accuracy"],
            metrics["expert_accuracy"],
        ]
        acc_colors = [PlotStyle.ACCENT, "#94a3b8", "#59A14F"]
        bars = axes[0].bar(acc_names, acc_values, color=acc_colors, alpha=0.9, edgecolor="white")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_title("System accuracy", fontsize=11, weight="bold", color="#1a4480")
        axes[0].set_ylabel("Test accuracy")
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        defer_names = ["Defer rate", "Precision", "Recall"]
        defer_values = [
            metrics["defer_rate"],
            metrics["defer_precision"],
            metrics["defer_recall"],
        ]
        bars2 = axes[1].bar(defer_names, defer_values, color=["#F28E2B", "#4E79A7", "#E15759"], alpha=0.9, edgecolor="white")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("Deferral quality", fontsize=11, weight="bold", color="#1a4480")
        axes[1].set_ylabel("Score")
        for bar in bars2:
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        for ax in axes:
            PlotStyle.apply(ax)
        fig.suptitle("Learning-to-defer evaluation", fontsize=12, weight="bold", color="#1a4480", y=1.02)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def active_curves(cls, entropy_steps, random_steps, baseline=None, oracle=None):
        fig, ax1 = plt.subplots(figsize=(8.5, 4.5), facecolor=PlotStyle.BG)
        x1 = [s.n_queries for s in entropy_steps]
        y1 = [s.team_accuracy for s in entropy_steps]
        x2 = [s.n_queries for s in random_steps]
        y2 = [s.team_accuracy for s in random_steps]

        ax1.plot(x1, y1, marker="o", markersize=4, linewidth=2, label="Hybrid strategy", color=PlotStyle.ACCENT)
        ax1.plot(x2, y2, marker="s", markersize=4, linewidth=2, label="Random baseline", color=PlotStyle.NEGATIVE)
        if baseline is not None:
            ax1.axhline(baseline, color="#94a3b8", linestyle="--", linewidth=1.5, label=f"Classifier only ({baseline:.3f})")
        if oracle is not None:
            ax1.axhline(oracle, color="#59A14F", linestyle=":", linewidth=1.5, label=f"Full rejector ({oracle:.3f})")

        ax1.set_xlabel("Expert queries")
        ax1.set_ylabel("Team accuracy (test)")
        ax1.set_title("Active learning progress", fontsize=12, weight="bold", color="#1a4480")
        ax1.legend(frameon=False, fontsize=8, loc="lower right")
        PlotStyle.apply(ax1)

        ax2 = ax1.twinx()
        ax2.plot(x1, [s.defer_rate for s in entropy_steps], color="#F28E2B", alpha=0.55, linewidth=1.5, linestyle="-.", label="Defer rate")
        ax2.set_ylabel("Defer rate", color="#F28E2B")
        ax2.tick_params(axis="y", labelcolor="#F28E2B", labelsize=9)
        ax2.spines["right"].set_visible(True)
        ax2.set_ylim(0, max(0.05, max((s.defer_rate for s in entropy_steps), default=0) * 1.2))

        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def human_vs_simulated(cls, report):
        per_class = report["per_class"]
        names = [n for n in CLASS_NAMES if per_class[n]["human"] is not None]
        if not names:
            return None
        fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=PlotStyle.BG)
        x = np.arange(len(names))
        width = 0.34
        human_vals = [per_class[n]["human"] for n in names]
        expert_vals = [per_class[n]["expert"] for n in names]
        ax.bar(x - width / 2, human_vals, width, label="You (human)", color="#E15759", alpha=0.9)
        ax.bar(x + width / 2, expert_vals, width, label="Simulated expert", color="#59A14F", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.05)
        ax.set_title("Human vs simulated expert (labeled articles)", fontsize=12, weight="bold", color="#1a4480")
        ax.set_ylabel("Accuracy")
        ax.legend(frameon=False, fontsize=9)
        PlotStyle.apply(ax)
        fig.tight_layout()
        return PlotStyle.save(fig)

    @classmethod
    def defer_inspector(cls, dataset, system, n=5):
        texts = dataset.test_texts[:200]
        labels = dataset.test_labels[:200]
        team, defer_mask = system.predict(texts)
        clf_preds = system.classifier.predict(texts)
        expert_preds = system.expert.predict_batch(texts)

        rows = []
        for i in range(len(texts)):
            rows.append(
                {
                    "deferred": bool(defer_mask[i]),
                    "team_ok": bool(team[i] == labels[i]),
                    "clf_ok": bool(clf_preds[i] == labels[i]),
                    "exp_ok": bool(expert_preds[i] == labels[i]),
                    "snippet": texts[i][:80] + "…",
                }
            )
        deferred = [r for r in rows if r["deferred"]][:n]
        kept = [r for r in rows if not r["deferred"]][:n]

        fig, ax = plt.subplots(figsize=(8, 4), facecolor=PlotStyle.BG)
        categories = ["Deferred (correct team)", "Deferred (wrong team)", "Kept (correct team)", "Kept (wrong team)"]
        counts = [
            sum(1 for r in deferred if r["team_ok"]),
            sum(1 for r in deferred if not r["team_ok"]),
            sum(1 for r in kept if r["team_ok"]),
            sum(1 for r in kept if not r["team_ok"]),
        ]
        colors = ["#59A14F", "#E15759", "#4E79A7", "#F28E2B"]
        bars = ax.bar(categories, counts, color=colors, alpha=0.9, edgecolor="white")
        ax.set_title("Defer decision quality (sample)", fontsize=11, weight="bold", color="#1a4480")
        ax.set_ylabel("Count (up to 5 each)")
        _label_bars(ax, bars, fmt="{:.0f}")
        PlotStyle.apply(ax)
        plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=8)
        fig.tight_layout()
        return PlotStyle.save(fig), {"deferred": deferred, "kept": kept}

    @staticmethod
    def per_class_accuracy(preds, labels):
        result = {}
        for idx, name in enumerate(CLASS_NAMES):
            mask = labels == idx
            if mask.sum() == 0:
                result[name] = 0.0
            else:
                result[name] = round(float(np.mean(preds[mask] == labels[mask])), 4)
        return result

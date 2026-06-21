import os
import uuid

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from django.conf import settings


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
        fig.patch.set_facecolor("#fafbfd")
        ax.set_facecolor("#fafbfd")
        target = dataset.target

        if dataset.problem_type == "regression" and not y_feature:
            ax.scatter(
                dataset.df[x_feature],
                target,
                alpha=0.75,
                s=48,
                c="#275CB2",
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

        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)

        return settings.MEDIA_URL + f"plots/{filename}"

import os
import uuid

import matplotlib.pyplot as plt
from django.conf import settings


class PlotBuilder:
    @classmethod
    def build(cls, dataset, x_feature, y_feature=None):
        if x_feature not in dataset.features:
            raise ValueError("Select a valid X feature.")

        plot_dir = os.path.join(settings.MEDIA_ROOT, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join(plot_dir, filename)

        plt.figure(figsize=(6, 4))
        target = dataset.target

        if dataset.problem_type == "regression" and not y_feature:
            plt.scatter(dataset.df[x_feature], target, alpha=0.7)
            plt.xlabel(x_feature)
            plt.ylabel(dataset.target_name)
            plt.title(f"{x_feature} vs {dataset.target_name}")
        else:
            if not y_feature or y_feature not in dataset.features:
                raise ValueError("Select two features for this plot.")
            labels = target.astype(str)
            for label in sorted(labels.unique()):
                mask = labels == label
                plt.scatter(
                    dataset.df.loc[mask, x_feature],
                    dataset.df.loc[mask, y_feature],
                    label=label,
                    alpha=0.7,
                )
            plt.xlabel(x_feature)
            plt.ylabel(y_feature)
            plt.title(f"{x_feature} vs {y_feature}")
            plt.legend(title=dataset.target_name)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        return settings.MEDIA_URL + f"plots/{filename}"

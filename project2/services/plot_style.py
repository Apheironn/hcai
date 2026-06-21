import os
import uuid

import matplotlib.pyplot as plt
from django.conf import settings


class PlotStyle:
    BG = "#fafbfd"
    ACCENT = "#275CB2"
    POSITIVE = "#275CB2"
    NEGATIVE = "#E15759"
    SPECIES_COLORS = {
        "Adelie": "#4E79A7",
        "Gentoo": "#59A14F",
        "Chinstrap": "#E15759",
    }
    DPI = 140

    @classmethod
    def apply(cls, ax):
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)

    @classmethod
    def save(cls, fig):
        plot_dir = os.path.join(settings.MEDIA_ROOT, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        filename = f"{uuid.uuid4()}.png"
        path = os.path.join(plot_dir, filename)
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return settings.MEDIA_URL + f"plots/{filename}"

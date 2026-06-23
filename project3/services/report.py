import os
import textwrap

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from django.conf import settings
from matplotlib.backends.backend_pdf import PdfPages

from .dataset import CLASS_NAMES


class ReportBuilder:
    LINE_WIDTH = 95

    def __init__(self, dataset, store_data):
        self.dataset = dataset
        self.data = store_data

    @classmethod
    def build(cls, dataset, store_data, output_path):
        builder = cls(dataset, store_data)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        builder._write(output_path)
        return output_path

    def _write(self, output_path):
        with PdfPages(output_path) as pdf:
            self._page(pdf, self._cover_lines())
            self._page(pdf, self._task1_lines())
            if self.data.get("expert_report"):
                self._page(pdf, self._task2_lines())
                self._plot_page(pdf, self.data.get("expert_chart_url"), "Expert vs classifier (test)")
            if self.data.get("defer_metrics"):
                self._page(pdf, self._task3_lines())
                self._plot_page(pdf, self.data.get("defer_chart_url"), "Learning-to-defer evaluation")
            if self.data.get("active_final_accuracy") is not None:
                self._page(pdf, self._task4_lines())
                self._plot_page(pdf, self.data.get("active_chart_url"), "Active learning curves")
            if self.data.get("human_report"):
                self._page(pdf, self._task5_lines())
                self._plot_page(pdf, self.data.get("human_chart_url"), "Human vs simulated expert")
            if self.data.get("defer_inspector_url"):
                self._plot_page(pdf, self.data.get("defer_inspector_url"), "Defer decision inspector")
            self._page(pdf, self._design_lines())

    def _page(self, pdf, lines):
        chunks = self._split_lines(lines)
        for chunk in chunks:
            fig = plt.figure(figsize=(8.5, 11), facecolor="white")
            y = 0.94
            for line in chunk:
                if line.startswith("## "):
                    fig.text(0.08, y, line[3:], fontsize=13, weight="bold", family="sans-serif")
                    y -= 0.035
                elif line == "":
                    y -= 0.012
                else:
                    for part in textwrap.wrap(line, width=self.LINE_WIDTH) or [""]:
                        fig.text(0.08, y, part, fontsize=10, family="sans-serif", va="top")
                        y -= 0.028
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    def _split_lines(self, lines):
        pages, current, y = [], [], 0.94
        for line in lines:
            extra = 0.035 if line.startswith("## ") else 0.012 if line == "" else 0.028 * max(1, len(textwrap.wrap(line, width=self.LINE_WIDTH)))
            if y - extra < 0.06 and current:
                pages.append(current)
                current, y = [], 0.94
            current.append(line)
            y -= extra
        if current:
            pages.append(current)
        return pages

    def _plot_page(self, pdf, plot_url, title):
        path = self._plot_path(plot_url)
        if not path or not os.path.isfile(path):
            return
        fig = plt.figure(figsize=(8.5, 11), facecolor="white")
        fig.text(0.08, 0.94, title, fontsize=13, weight="bold")
        img = mpimg.imread(path)
        ax = fig.add_axes([0.08, 0.12, 0.84, 0.76])
        ax.imshow(img)
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def _plot_path(self, plot_url):
        if not plot_url:
            return None
        marker = "/plots/"
        if marker not in plot_url:
            return None
        filename = plot_url.split(marker, 1)[1]
        return os.path.join(settings.MEDIA_ROOT, "plots", filename)

    def _cover_lines(self):
        return [
            "## Project 3 Report — Learning to Defer",
            "",
            "Course: Human-Centric Artificial Intelligence",
            "Dataset: AG News (fancyzhx/ag_news)",
            f"Train samples: {self.dataset.n_train}",
            f"Test samples: {self.dataset.n_test}",
            f"Classes: {', '.join(CLASS_NAMES)}",
            "",
            "This report summarizes the experiments run in the Project 3 interface.",
        ]

    def _task1_lines(self):
        acc = self.data.get("baseline_accuracy")
        return [
            "## Task 1 — Baseline Classifier",
            "",
            "Model: TF-IDF (8000 features, unigrams + bigrams) + Logistic Regression",
            "Training: full AG News train split",
            "Evaluation: held-out test split",
            "",
            f"Test accuracy: {acc if acc is not None else 'not available'}",
        ]

    def _task2_lines(self):
        report = self.data["expert_report"]
        lines = [
            "## Task 2 — Simulated Expert",
            "",
            f"Expert: {report['name']}",
            f"Overall test accuracy: {report['overall_accuracy']}",
            "",
            "Per-class test accuracy:",
        ]
        for name, acc in report["per_class_accuracy"].items():
            lines.append(f"  • {name}: {acc}")
        lines.extend([
            "",
            f"Strengths: {report['strengths']}",
            f"Weaknesses: {report['weaknesses']}",
        ])
        return lines

    def _task3_lines(self):
        m = self.data["defer_metrics"]
        return [
            "## Task 3 — Learning to Defer",
            "",
            "Strategy: binary rejector (defer when expert is correct and classifier is wrong)",
            "Rejector features: same TF-IDF vectorizer as the baseline classifier",
            "",
            f"Team accuracy: {m['team_accuracy']}",
            f"Classifier only: {m['classifier_accuracy']}",
            f"Expert only: {m['expert_accuracy']}",
            f"Defer rate: {m['defer_rate']}",
            f"Defer precision: {m['defer_precision']}",
            f"Defer recall: {m['defer_recall']}",
        ]

    def _task4_lines(self):
        lines = [
            "## Task 4 — Active Learning",
            "",
            "Setting: classifier trained on full labels; expert labels collected via queries on train only.",
            "Strategy: entropy sampling for the first half of the budget, then rejector uncertainty.",
            "Baseline: uniform random querying on the same budget.",
            "",
        ]
        if self.data.get("active_budget"):
            lines.append(f"Query budget: {self.data['active_budget']}")
        if self.data.get("active_batch_size"):
            lines.append(f"Batch size: {self.data['active_batch_size']}")
        lines.extend([
            "",
            f"Final team accuracy (hybrid): {self.data.get('active_final_accuracy')}",
            f"Final team accuracy (random): {self.data.get('active_random_final_accuracy')}",
        ])
        oracle = self.data.get("defer_metrics", {}).get("team_accuracy")
        if oracle is not None:
            lines.append(f"Task 3 oracle (full rejector): {oracle}")
        return lines

    def _task5_lines(self):
        report = self.data["human_report"]
        lines = [
            "## Task 5 — Human Expert Interaction",
            "",
            f"Articles labeled by user: {report['n_labeled']}",
            f"Human accuracy on labeled subset: {report['human_accuracy']}",
            f"Simulated expert on same subset: {report['expert_accuracy']}",
            "",
            "Per-class comparison on labeled articles:",
        ]
        for name, vals in report["per_class"].items():
            if vals["human"] is None:
                continue
            lines.append(f"  • {name}: human={vals['human']}, simulated={vals['expert']}")
        lines.append("")
        lines.append(
            "Interpretation: comparing your labels to the keyword expert shows where "
            "human judgment differs from the simulated regional specialist."
        )
        return lines

    def _design_lines(self):
        return [
            "## Design Choices",
            "",
            "• Baseline: TF-IDF + logistic regression for a strong, interpretable text benchmark.",
            "• Expert: keyword-based regional specialist (strong on Sports/Business, weaker elsewhere).",
            "• Deferral: rejector trained on cases where the expert is correct and the classifier is wrong.",
            "• Active learning: first half of budget uses entropy on classifier uncertainty; "
            "second half queries points where the rejector is uncertain (near 0.5 defer probability).",
            "• Random baseline uses the same query budget for fair comparison.",
            "• Human expert UI collects real labels on train articles to compare against simulation.",
        ]

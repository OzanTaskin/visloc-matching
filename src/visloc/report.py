"""Turn per-run metrics into results.csv, a LaTeX table and figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import CONDITIONS, METHODS  # noqa: E402

RECALL_FINE = "recall_25cm_2deg"
RECALL_COARSE = "recall_500cm_10deg"


def write_tables(summary: pd.DataFrame, results_dir: Path) -> None:
    summary.to_csv(results_dir / "results.csv", index=False, float_format="%.2f")
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Condition & Method & Med.\ $t$ (cm) & Med.\ $R$ (\textdegree) "
        r"& 25\,cm,2\textdegree & 50\,cm,5\textdegree & 5\,m,10\textdegree & s/query \\",
        r"\midrule",
    ]
    for cond in CONDITIONS:
        rows = summary[summary.condition == cond]
        for i, r in enumerate(rows.itertuples()):
            first = cond if i == 0 else ""
            med_t = "fail" if np.isinf(r.median_t_cm) else f"{r.median_t_cm:.1f}"
            lines.append(
                f"{first} & {METHODS[r.method].label} & {med_t} & {r.median_R_deg:.2f} & "
                f"{r.recall_25cm_2deg:.1f} & {r.recall_50cm_5deg:.1f} & "
                f"{r.recall_500cm_10deg:.1f} & {r.seconds_per_query:.2f} \\\\"
            )
        lines.append(r"\midrule" if cond != list(CONDITIONS)[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (results_dir / "table.tex").write_text("\n".join(lines) + "\n")


def plot_recall(summary: pd.DataFrame, out: Path) -> None:
    conds, methods = list(CONDITIONS), [m for m in METHODS if m in set(summary.method)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    width = 0.8 / len(methods)
    for ax, key, title in [(axes[0], RECALL_FINE, "Fine: within 25 cm and 2°"),
                           (axes[1], RECALL_COARSE, "Coarse: within 5 m and 10°")]:
        for i, m in enumerate(methods):
            vals = [summary[(summary.method == m) & (summary.condition == c)][key].max()
                    for c in conds]
            ax.bar(np.arange(len(conds)) + i * width, vals, width, label=METHODS[m].label)
        ax.set_xticks(np.arange(len(conds)) + width * (len(methods) - 1) / 2, conds)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Queries localized (%)")
    axes[1].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_cdf(per_query: pd.DataFrame, out: Path) -> None:
    conds = list(CONDITIONS)
    fig, axes = plt.subplots(1, len(conds), figsize=(13, 3.2), sharey=True)
    for ax, cond in zip(axes, conds):
        for m in METHODS:
            err = per_query[(per_query.method == m) & (per_query.condition == cond)].err_t_m
            if err.empty:
                continue
            err = np.sort(err.to_numpy() * 100)
            ax.plot(err, np.arange(1, len(err) + 1) / len(err) * 100, label=METHODS[m].label)
        ax.set_xscale("log")
        ax.set_xlim(1, 1000)
        ax.set_title(cond)
        ax.set_xlabel("Position error (cm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Queries (%)")
    axes[-1].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def make_all(summary: pd.DataFrame, per_query: pd.DataFrame, results_dir: Path) -> None:
    figures = results_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    order = {c: i for i, c in enumerate(CONDITIONS)} | {m: i for i, m in enumerate(METHODS)}
    summary = summary.sort_values(["condition", "method"], key=lambda s: s.map(order))
    write_tables(summary, results_dir)
    per_query.to_csv(results_dir / "per_query.csv", index=False, float_format="%.4f")
    plot_recall(summary, figures / "recall.png")
    plot_cdf(per_query, figures / "error_cdf.png")

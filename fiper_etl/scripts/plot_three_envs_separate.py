"""
plot_three_envs_separate.py
---------------------------
Three separate PDFs comparing ETL vs FIPER baselines (PCA-kmeans, logpZO) on
Stacking / Sorting / MetaWorld pick-place-wall:

  1. f1.pdf            — F1 at *both* operating points (CP + best, paired bars).
  2. accuracy_best.pdf — Accuracy at the F1-optimal threshold.
  3. recall_cp.pdf     — Recall at the calibrated (CP) threshold.

Color scheme matches the user's reference: pastel pink/firebrick + pastel
orange/darkorange, extended with pastel blue/navy for the third method.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
NEWT_ROOT = ROOT.parent
OUT_DIR = NEWT_ROOT / "figures" / "fiper_three_envs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "data" / "results" / "etl_f1_results.json") as f:
    fiper = json.load(f)
with open(NEWT_ROOT / "etl_results" / "mw_fiper_baselines" / "fiper_metrics.json") as f:
    mw = json.load(f)["results"]


def mw_avg(method_mw: str, key: str) -> float:
    return 0.5 * (mw[method_mw]["subtask_A"][key] + mw[method_mw]["subtask_B"][key])


# Single source of truth for what to fetch per (env, metric)
def get(env_key: str, m_fiper: str, m_mw: str, metric: str) -> float:
    if env_key == "mw_pick_place_wall":
        return {
            "f1_cp":             lambda: mw_avg(m_mw, "cp_f1"),
            "f1_optimal":        lambda: mw_avg(m_mw, "f1_f1"),
            "accuracy_at_f1opt": lambda: mw_avg(m_mw, "f1_agreement"),
            "recall_cp":         lambda: mw_avg(m_mw, "cp_recall"),
            "recall_at_f1opt":   lambda: mw_avg(m_mw, "f1_recall"),
        }[metric]()
    return fiper[env_key][m_fiper][metric]


ENVS = [
    ("Stacking",                       "stacking"),
    ("Sorting",                        "sorting"),
    ("MW-Pick-Place\n(grasp ∧ place)", "mw_pick_place_wall"),
]

# (label, fiper-key, mw-key, fill, edge)
METHODS = [
    ("ETL",        "ETL-or",     "etl",        "#FFD580", "darkorange"),  # pastel orange / "ours"
    ("PCA-kmeans", "PCA-kmeans", "pca_kmeans", "#FFCCCB", "firebrick"),   # pastel pink
    ("logpZO",     "logpZO",     "logpzo",     "#B0D8FF", "navy"),        # pastel blue
]

n_envs    = len(ENVS)
n_methods = len(METHODS)


TITLE_FS = 14
LABEL_FS = 12
NUM_FS   = 10
TICK_FS  = 11

plt.rcParams.update({
    "font.size":       11,
    "axes.labelsize":  LABEL_FS,
    "xtick.labelsize": TICK_FS,
    "ytick.labelsize": TICK_FS,
    "legend.fontsize": 11,
})


def annotate(ax, bars, vals, edge):
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
            f"{v:.3f}", ha="center", va="bottom",
            fontsize=NUM_FS, fontweight="bold", color=edge,
        )


# ────────────────────────────────────────────────────────────────────────────
# Plot 1 — F1: best operating point for everyone, plus CP-calibrated for ETL.
# (CP isn't a fair operating point for logpZO / PCA-kmeans — their thresholds
# aren't designed for class-conditional CP, so we report them at best F1 only.)
# ────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
bar_w = 0.22
group_gap = 0.12
group_w = n_methods * bar_w
x = np.arange(n_envs) * (group_w + group_gap)

for j, (m_label, m_fiper, m_mw, fill, edge) in enumerate(METHODS):
    vals = [get(e, m_fiper, m_mw, "f1_optimal") for _, e in ENVS]
    bars = ax.bar(
        x + j * bar_w, vals, bar_w,
        color=fill, edgecolor=edge, linewidth=1.4, label=m_label,
    )
    annotate(ax, bars, vals, edge)

ax.set_xticks(x + group_w / 2 - bar_w / 2)
ax.set_xticklabels([e[0] for e in ENVS], fontsize=TICK_FS, fontweight="bold")
ax.set_ylabel("F1", fontsize=LABEL_FS, fontweight="bold")
ax.set_xlabel("Environment", fontsize=LABEL_FS, fontweight="bold")
ax.set_title("F1 scores for monitoring: ETL vs logpZO and kmeans", fontsize=TITLE_FS, fontweight="bold")
ax.set_ylim(0, 1.22)
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.grid(axis="y", linewidth=0.5, alpha=0.4)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=11, ncol=3, loc="upper center", frameon=True)
plt.tight_layout()
fig.savefig(OUT_DIR / "f1.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "f1.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ────────────────────────────────────────────────────────────────────────────
# Helper — single-metric grouped bar chart
# ────────────────────────────────────────────────────────────────────────────
def single_metric_plot(
    metric_key: str,
    title: str,
    ylabel: str,
    out_stem: str,
    methods: list = METHODS,
) -> None:
    n_m = len(methods)
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_w_local = 0.22
    x_local = np.arange(n_envs) * (n_m * bar_w_local + 0.10)

    for j, (m_label, m_fiper, m_mw, fill, edge) in enumerate(methods):
        vals = [get(e, m_fiper, m_mw, metric_key) for _, e in ENVS]
        bars = ax.bar(
            x_local + j * bar_w_local, vals, bar_w_local,
            color=fill, edgecolor=edge, linewidth=1.4, label=m_label,
        )
        annotate(ax, bars, vals, edge)

    ax.set_xticks(x_local + (n_m - 1) * bar_w_local / 2)
    ax.set_xticklabels([e[0] for e in ENVS], fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_xlabel("Environment", fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


single_metric_plot(
    "accuracy_at_f1opt",
    "Accuracy at F1-optimal threshold",
    "Accuracy",
    "accuracy_best",
)


# ────────────────────────────────────────────────────────────────────────────
# Plot 3 — Recall: ETL at CP threshold, baselines at F1-optimal threshold.
# (CP isn't a meaningful operating point for logpZO / PCA-kmeans, so we report
# their best-case recall — the recall they achieve at the F1-maximising tau.)
# ────────────────────────────────────────────────────────────────────────────
def recall_plot() -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_w_local = 0.22
    x_local = np.arange(n_envs) * (n_methods * bar_w_local + 0.10)

    for j, (m_label, m_fiper, m_mw, fill, edge) in enumerate(METHODS):
        metric_key = "recall_cp" if m_label == "ETL" else "recall_at_f1opt"
        vals = [get(e, m_fiper, m_mw, metric_key) for _, e in ENVS]
        legend_label = (
            f"{m_label} (CP)" if m_label == "ETL" else f"{m_label}"
        )
        bars = ax.bar(
            x_local + j * bar_w_local, vals, bar_w_local,
            color=fill, edgecolor=edge, linewidth=1.4, label=legend_label,
        )
        annotate(ax, bars, vals, edge)

    ax.set_xticks(x_local + (n_methods - 1) * bar_w_local / 2)
    ax.set_xticklabels([e[0] for e in ENVS], fontsize=12, fontweight="bold")
    ax.set_ylabel("Recall", fontsize=14, fontweight="bold")
    ax.set_xlabel("Environment", fontsize=14, fontweight="bold")
    ax.set_title("Recall (ETL @ CP threshold; baselines @ best-F1 threshold)",
                 fontsize=15, fontweight="bold")
    ax.set_ylim(0, 1.10)
    ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(axis="y", linewidth=0.5, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "recall_cp.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / "recall_cp.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


recall_plot()

# ── console summary ──
print(f"Wrote PDFs / PNGs to {OUT_DIR}/")


# ────────────────────────────────────────────────────────────────────────────
# F1 table (LaTeX + ASCII).  Rows: methods (ETL split into best/CP);
# Cols: environments + Avg.  Bolded per column = best F1.
# ────────────────────────────────────────────────────────────────────────────
ENV_TEX_HEADERS = ["Stacking", "Sorting", r"MW-Pick-Place"]

# (display_label, getter)
ROWS = [
    ("logpZO",       lambda env_key: get(env_key, "logpZO",     "logpzo",     "f1_optimal")),
    ("PCA-kmeans",   lambda env_key: get(env_key, "PCA-kmeans", "pca_kmeans", "f1_optimal")),
    ("ETL (best F1)", lambda env_key: get(env_key, "ETL-or",    "etl",        "f1_optimal")),
    ("ETL (CP)",     lambda env_key: get(env_key, "ETL-or",    "etl",        "f1_cp")),
]

env_keys = [e[1] for e in ENVS]
matrix = [[fn(k) for k in env_keys] for _, fn in ROWS]
avgs = [float(np.mean(row)) for row in matrix]

# Per-column best (compared across all rows so ETL-best vs ETL-CP both fight)
col_best_idx = [int(np.argmax([row[c] for row in matrix])) for c in range(len(env_keys))]
avg_best_idx = int(np.argmax(avgs))


def fmt(v: float, bold: bool) -> str:
    s = f"{v:.3f}"
    return rf"\textbf{{{s}}}" if bold else s


tex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{F1 score for failure / predicate detection across three environments. "
    r"All methods use the F1-optimal threshold; ETL (CP) additionally reports the "
    r"class-conditional split-CP threshold. Per-column best in \textbf{bold}.}",
    r"\label{tab:f1_three_envs}",
    r"\begin{tabular}{l" + "c" * (len(env_keys) + 1) + r"}",
    r"\toprule",
    r"Method & " + " & ".join(ENV_TEX_HEADERS) + r" & Avg \\",
    r"\midrule",
]
for r_idx, (label, _) in enumerate(ROWS):
    cells = [
        fmt(matrix[r_idx][c], col_best_idx[c] == r_idx)
        for c in range(len(env_keys))
    ]
    cells.append(fmt(avgs[r_idx], avg_best_idx == r_idx))
    tex_lines.append(label + " & " + " & ".join(cells) + r" \\")
tex_lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
]
tex = "\n".join(tex_lines) + "\n"

tex_path = OUT_DIR / "f1_table.tex"
tex_path.write_text(tex)

# ── ASCII echo ──
col_w = 14
print()
print(f"F1 table → {tex_path}")
print()
print(f"{'Method':<14}", end="")
for h in ENV_TEX_HEADERS + ["Avg"]:
    print(f"{h:>{col_w}}", end="")
print()
print("-" * (14 + col_w * (len(env_keys) + 1)))
for r_idx, (label, _) in enumerate(ROWS):
    print(f"{label:<14}", end="")
    for c in range(len(env_keys)):
        marker = "*" if col_best_idx[c] == r_idx else " "
        print(f"{matrix[r_idx][c]:>{col_w-1}.3f}{marker}", end="")
    marker = "*" if avg_best_idx == r_idx else " "
    print(f"{avgs[r_idx]:>{col_w-1}.3f}{marker}")
print("\n(* marks per-column best — bolded in LaTeX.)")
print()
print(tex)

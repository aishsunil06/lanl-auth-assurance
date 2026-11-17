"""Charts for the audit results. Usage: python src/plots.py"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#1b1b1b"
MUTED = "#9a9a9a"
ACCENT = "#c1432c"
BLUE = "#2e5c8a"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)


def fig1_split_protocol(R, path):
    """The same model and features, scored under two evaluation protocols."""
    labels = ["Random split\n(standard practice)"]
    vals = [R["E2_random_split"]["roc_auc"]]
    colors = [ACCENT]
    for key, v in R["E2_temporal_sweep"].items():
        a = key.replace("train_", "").replace("_test_", "  ->  ")
        labels.append(f"Temporal\n{a}")
        vals.append(v["roc_auc"])
        colors.append(BLUE)

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    bars = ax.bar(labels, vals, color=colors, width=0.6)
    ax.axhline(0.5, ls="--", lw=1, color=MUTED)
    ax.text(len(vals) - 0.4, 0.515, "chance", fontsize=8, color=MUTED, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=9, weight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("The same detector, scored two ways",
                 loc="left", fontsize=11, weight="bold")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig2_concentration(R, path):
    """Why the benchmark cannot test generalisation across attacker hosts."""
    e1 = R["E1_held_out_attacker_host"]
    dom = e1["redteam_hops_from_dominant"]
    oth = e1["redteam_hops_from_others"]

    fig, ax = plt.subplots(figsize=(6.6, 1.9))
    ax.barh([""], [dom], color=ACCENT, height=0.45,
            label=f"{e1['dominant_host']}  ({dom} hops)")
    ax.barh([""], [oth], left=[dom], color=BLUE, height=0.45,
            label=f"all other hosts  ({oth} hops)")
    ax.set_xlim(0, dom + oth)
    ax.set_xlabel("Red team hops by originating host")
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.45), ncol=2)
    ax.set_title(
        f"{e1['dominant_share']:.0%} of attack traffic comes from one host",
        loc="left", fontsize=11, weight="bold")
    ax.spines["left"].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig3_adversary(R, path):
    e3 = R.get("E3_adaptive_adversary", {})
    if "note" in e3:
        return False
    order = [
        ("unmodified", "Real red team activity"),
        ("no_failed_attempts", "Makes no failed attempts"),
        ("business_hours", "Moves during business hours"),
        ("familiar_destination", "Targets a host the user already uses"),
        ("familiar_source_path", "Arrives from a familiar source"),
        ("blend_into_volume", "Pads session to normal volume"),
        ("combined", "All of the above"),
    ]
    rows = [(lab, e3[k]["1000"]["recall"]) for k, lab in order if k in e3]
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [BLUE] * len(vals)
    colors[-1] = MUTED
    colors[0] = ACCENT

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    bars = ax.barh(labels, vals, color=colors, height=0.6)
    for b, v in zip(bars, vals):
        ax.text(v + 0.015, b.get_y() + b.get_height() / 2, f"{v:.0%}",
                va="center", fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Recall at an alert budget of 1000")
    ax.set_title("Detection lost to an intruder with better habits",
                 loc="left", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    with open("results/audit.json") as fh:
        R = json.load(fh)
    fig1_split_protocol(R, "results/fig1_split_protocol.png")
    fig2_concentration(R, "results/fig2_concentration.png")
    ok3 = fig3_adversary(R, "results/fig3_adversary.png")
    print(f"wrote fig1, fig2, fig3={ok3}")


if __name__ == "__main__":
    main()

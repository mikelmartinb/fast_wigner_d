"""Plot benchmarks/results/results.json as benchmark.png (light) and benchmark-dark.png."""
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"

THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]),
    "dark": dict(surface="#1a1a19", primary="#f0efec", secondary="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]),
}
# fixed colour per method (the 1-thread run shares it, dashed)
SLOT = {"fast_wigner_d": 0, "fast_wigner_d_1thread": 0, "spherical": 1, "wigner": 2,
        "expm": 3, "eval_jacobi": 4, "textbook": 5, "sympy": 6}
USABLE = 1e-8          # larger errors (or NaN) count as a wrong result
ERR_FLOOR, ERR_FAIL = 1e-17, 3.0


def is_wrong(r):
    return math.isnan(r["error"]) or r["error"] > USABLE


def dodge(points, factor=1.13):
    """Spread crosses sharing an x position (log axis), after merging a method's
    neighbouring half-integer and integer sizes (j = 49.5 and 50) into one."""
    merged = []
    for slot, x, y in sorted(points):
        if merged and merged[-1][0] == slot and x < merged[-1][1] * 1.05:
            continue
        merged.append((slot, x, y))
    by_x = {}
    for slot, x, y in merged:
        x = math.floor(x + 0.5) or x  # j = 999.5 and j = 1000 share a column
        by_x.setdefault(x, []).append((slot, y))
    for x, group in by_x.items():
        group.sort()
        for k, (slot, y) in enumerate(group):
            yield slot, x * factor ** (k - (len(group) - 1) / 2), y


def plot(data, theme, path):
    t = THEMES[theme]
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 9.5})
    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(11, 4.9), dpi=160)
    fig.patch.set_facecolor(t["surface"])
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.27, wspace=0.22)

    rows, overflowed = {}, {}
    for r in data["results"]:
        if r["status"] == "ok":
            rows.setdefault(r["method"], []).append(r)
        elif r["status"] == "failed":
            overflowed.setdefault(r["method"], []).append(r)

    handles, crosses_t, crosses_e = [], [], []
    for name in sorted(rows, key=lambda n: (SLOT[n], n)):
        rs = sorted(rows[name], key=lambda r: r["two_j"])
        color = t["series"][SLOT[name]]
        one_thread = name.endswith("1thread")
        line = dict(color=color, lw=1.8, ls=(0, (4, 2)) if one_thread else "-",
                    marker="o", ms=4.5, mew=1.0, solid_capstyle="round",
                    mfc=t["surface"] if one_thread else color,
                    mec=color if one_thread else t["surface"])
        good = [r for r in rs if not is_wrong(r)]
        ax_t.plot([r["two_j"] / 2 for r in good], [r["time"] for r in good], **line)
        crosses_t += [(SLOT[name], r["two_j"] / 2, r["time"]) for r in rs if is_wrong(r)]
        if not one_thread:  # same numbers as the threaded run
            finite = [r for r in rs if not math.isnan(r["error"]) and r["error"] < 1]
            ax_e.plot([r["two_j"] / 2 for r in finite],
                      [max(r["error"], ERR_FLOOR) for r in finite], **line)
            failed = [r for r in rs if r not in finite] + overflowed.get(name, [])
            crosses_e += [(SLOT[name], r["two_j"] / 2, ERR_FAIL) for r in failed]
        handles.append(Line2D([], [], **line, label=data["methods"][name]["label"]))
    for ax, crosses in [(ax_t, crosses_t), (ax_e, crosses_e)]:
        for slot, x, y in dodge(crosses):
            ax.plot([x], [y], ls="none", marker="x", ms=6, mew=1.6, color=t["series"][slot])
    handles.append(Line2D([], [], ls="none", marker="x", ms=6, mew=1.6, color=t["secondary"],
                          label=f"inaccurate or failed (error > {USABLE:.0e}, NaN, overflow)"))

    for ax, title in [(ax_t, "Time per full matrix (s) — lower is better"),
                      (ax_e, "Largest absolute error vs. mpmath — lower is better")]:
        ax.set_facecolor(t["surface"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(title, loc="left", color=t["primary"], fontsize=10.5, pad=8)
        ax.set_xlabel("spin j  (integer and half-integer)", color=t["secondary"])
        ax.xaxis.set_major_locator(FixedLocator([0.5, 1, 10, 100, 1000]))
        ax.set_xticklabels(["½", "1", "10", "100", "1000"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", color=t["grid"], lw=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(t["axis"])
        ax.tick_params(which="both", colors=t["axis"], labelcolor=t["secondary"])
        ax.set_xlim(0.4, 2600)

    ax_t.yaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax_e.set_ylim(ERR_FLOOR / 2, 10)
    ax_e.yaxis.set_major_locator(FixedLocator([1e-16, 1e-13, 1e-10, 1e-7, 1e-4, 1e-1]))
    ax_e.axhline(1, color=t["axis"], lw=0.8)
    ax_e.text(0.45, ERR_FAIL, "failed", color=t["muted"], va="center", fontsize=8.5)

    m = data["machine"]
    fig.text(0.07, 0.95, "Full Wigner d-matrix, all (2j+1)² elements per call",
             color=t["primary"], fontsize=13, weight="bold")
    fig.text(0.07, 0.905, f"{m['cpu']}, {m['cores']} cores · Python {m['python']} · "
             f"median over β = 0.3, 1.1, 2.6 · errors below 1e-17 drawn at 1e-17",
             color=t["muted"], fontsize=9)
    legend = fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
                        bbox_to_anchor=(0.52, 0.0), fontsize=9, handlelength=2.6,
                        columnspacing=1.6)
    for text in legend.get_texts():
        text.set_color(t["secondary"])
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)


def main():
    data = json.loads((RESULTS / "results.json").read_text())
    plot(data, "light", RESULTS / "benchmark.png")
    plot(data, "dark", RESULTS / "benchmark-dark.png")
    print("wrote", RESULTS / "benchmark.png", "and", RESULTS / "benchmark-dark.png")


if __name__ == "__main__":
    main()

import numpy as np

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def rolling(a, window):
    # trailing mean over full windows, value i ends at index i + window - 1
    c = np.concatenate([[0.0], np.cumsum(a)])
    return (c[window:] - c[:-window]) / window


def panels(heights, height=6.4):
    """a figure of stacked panels sharing x, one per entry of heights, as (fig, axes)."""
    # imported here, code that only computes numbers doesn't pay for matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt.subplots(len(heights), 1, figsize=(9, height), sharex=True, facecolor=SURFACE,
                        gridspec_kw={"height_ratios": heights})


def panel(ax, x, lines, name):
    """draw lines, each (label, values, color, width), on ax and call the axis name."""
    for label, values, color, width in lines:
        ax.plot(x, values, color=color, linewidth=width, label=label)
    style(ax, name)
    if len(lines) > 1:
        legend(ax, len(lines))
    if 1 < len(lines) <= 4:
        _label_ends(ax, x[-1], [(values[-1], label) for label, values, _, _ in lines])


def style(ax, name):
    """the house look for ax, its y axis called name."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_ylabel(name, color=MUTED)


def legend(ax, columns):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), frameon=False, fontsize=9,
              labelcolor=INK, ncol=columns)


def _label_ends(ax, x, ends):
    # push labels apart so they don't overlap, at least 6% of the axis height
    lo, hi = ax.get_ylim()
    gap = 0.06 * (hi - lo)
    placed = []
    for y, label in sorted(ends):
        y = max(y, placed[-1] + gap) if placed else y
        placed.append(y)
        ax.annotate(label, (x, y), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK, fontsize=8, annotation_clip=False)


def mark(axes, step, text):
    """a vertical rule at step on every panel, with text at the top of the first."""
    for ax in axes:
        ax.axvline(step, color=AXIS, linewidth=1, zorder=1)
    axes[0].annotate(text, (step, 1), xycoords=("data", "axes fraction"), xytext=(4, -12),
                     textcoords="offset points", color=MUTED, fontsize=9)


def save(fig, title, path):
    import matplotlib.pyplot as plt

    fig.suptitle(title, color=INK, fontsize=10, x=0.06, ha="left")
    fig.tight_layout()
    fig.subplots_adjust(right=0.8)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)

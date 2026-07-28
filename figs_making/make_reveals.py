#!/usr/bin/env python3
"""Regenerate reveal1_matched_collapse and reveal3_identity_vs_prediction
(png + pdf + transparent png), with the frozen-protocol numbers:
descriptor .707 -> .550, learned .935 -> .811, and 'conditions' terminology."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = "./figs_making"
BLUE = "#7189b9"      # learned
TEAL = "#45a49b"      # descriptor
INK = "#3d4451"
GRAY = "#8a93a5"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def save(fig, stem):
    for ext, kw in (("png", {}), ("pdf", {}), ("_transparent.png", {"transparent": True})):
        p = f"{HERE}/{stem}{'.' + ext if not ext.startswith('_') else ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", **kw)
    print("saved", stem)


# ---------- reveal 1: condition-matched collapse (slope chart) ----------
fig, ax = plt.subplots(figsize=(4.8, 3.55))
x = [0, 1]
learned = [0.935, 0.811]
desc = [0.707, 0.550]
ax.plot(x, learned, "-o", color=BLUE, lw=3.5, ms=10, zorder=3)
ax.plot(x, desc, "-o", color=TEAL, lw=3.5, ms=10, zorder=3)
ax.axhline(0.5, color=GRAY, ls=":", lw=1.8)
ax.text(0.5, 0.508, "chance", color=GRAY, fontsize=12, ha="center")
ax.text(-0.09, learned[0], "Learned", color=BLUE, fontsize=14, fontweight="bold",
        ha="right", va="center")
ax.text(-0.09, desc[0] - 0.028, "Descriptor", color=TEAL, fontsize=14, fontweight="bold",
        ha="right", va="center")
for xi, yi in zip(x, learned):
    ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                xytext=(14 if xi else -6, 14), ha="center", fontsize=13, color=INK)
for xi, yi in zip(x, desc):
    ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                xytext=(14 if xi else -6, 14), ha="center", fontsize=13, color=INK)
ax.set_xlim(-0.78, 1.38)
ax.set_ylim(0.47, 1.02)
ax.set_xticks(x)
ax.set_xticklabels(["Unmatched\nconditions", "Matched\nconditions"], fontsize=14, color=INK)
ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_ylabel("driver re-ID AUROC", fontsize=14, color=INK)
ax.tick_params(colors=INK, labelsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(INK)
save(fig, "reveal1_matched_collapse")
plt.close(fig)

# ---------- reveal 3: identity vs prediction (two stat boxes) ----------
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(5.2, 2.1))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(cx, color, title, stat):
    ax.add_patch(FancyBboxPatch((cx - 0.215, 0.10), 0.43, 0.80,
                                boxstyle="round,pad=0.012,rounding_size=0.03",
                                fill=False, edgecolor=color, lw=2.4,
                                transform=ax.transAxes, clip_on=False))
    ax.text(cx, 0.70, title, ha="center", va="center", fontsize=9.5, color=INK,
            transform=ax.transAxes)
    ax.text(cx, 0.33, stat, ha="center", va="center", fontsize=12.5, color=color,
            fontweight="bold", transform=ax.transAxes)


box(0.23, TEAL, "Best re-ID embedding\nas conditioning", "PG  −0.2%")
box(0.77, BLUE, "Task-aligned few-shot\nstyle via FiLM", "PG  +0.4–1.4%")
ax.text(0.5, 0.5, "vs.", ha="center", va="center", fontsize=11, color=GRAY,
        style="italic", transform=ax.transAxes)
save(fig, "reveal3_identity_vs_prediction")
plt.close(fig)

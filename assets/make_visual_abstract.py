"""Generate the repository visual abstract (assets/visual_abstract.svg).

The covariance heatmap is *real* library output: a fixed 2-dimensional diamond
DAG is built with :class:`gaussian_bn.GaussianDAG` and its full covariance is
computed by ``k_full`` (the K-recursion). Re-run this script to regenerate the
figure deterministically:

    uv run python assets/make_visual_abstract.py
"""

from __future__ import annotations

import pathlib

import torch

import gaussian_bn as gbn

# ---------------------------------------------------------------------------
# The model behind the heatmap: a vector-valued (2-dim nodes) diamond DAG,
# fixed parameters (no randomness) so the figure is exactly reproducible.
# ---------------------------------------------------------------------------
T = lambda rows: torch.tensor(rows, dtype=torch.float64)  # noqa: E731

MODEL = gbn.GaussianDAG(
    dims=[2, 2, 2, 2],
    edges={
        (0, 1): T([[1.1, 0.4], [-0.3, 0.8]]),
        (0, 2): T([[-0.7, 0.2], [0.5, -0.9]]),
        (1, 3): T([[0.9, -0.2], [0.3, 0.7]]),
        (2, 3): T([[0.6, 0.3], [-0.4, 1.0]]),
    },
    noise=[
        T([[1.0, 0.2], [0.2, 0.8]]),
        T([[0.4, 0.0], [0.0, 0.4]]),
        T([[0.5, -0.1], [-0.1, 0.5]]),
        T([[0.3, 0.0], [0.0, 0.3]]),
    ],
)

# -- palette ----------------------------------------------------------------
INK = "#1e293b"        # main text
MUTED = "#64748b"      # secondary text
CARD_BG = "#f8fafc"
CARD_EDGE = "#e2e8f0"
NODE_FILL = "#dbeafe"
NODE_EDGE = "#1d4ed8"
ACCENT = "#d97706"     # autograd badge
POS = (29, 78, 216)    # heatmap positive -> blue
NEG = (220, 38, 38)    # heatmap negative -> red

FONT = "Helvetica, Arial, sans-serif"


def cell_color(v: float, vmax: float) -> str:
    """White -> blue for positive, white -> red for negative entries."""
    t = max(-1.0, min(1.0, v / vmax))
    base = POS if t >= 0 else NEG
    a = abs(t) ** 0.7                       # gamma lift so small values stay visible
    r, g, b = (round(255 + (c - 255) * a) for c in base)
    return f"rgb({r},{g},{b})"


def sub(main: str, subscript: str, *, x: float, y: float, size: int,
        fill: str = INK, anchor: str = "middle", style: str = "italic") -> str:
    """Text with a subscript tspan."""
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-style="{style}" fill="{fill}" text-anchor="{anchor}">{main}'
        f'<tspan font-size="{round(size * 0.72)}" baseline-shift="-22%">{subscript}</tspan></text>'
    )


def text(s: str, *, x: float, y: float, size: int, fill: str = INK,
         anchor: str = "middle", weight: str = "normal", style: str = "normal") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" font-style="{style}" fill="{fill}" '
        f'text-anchor="{anchor}">{s}</text>'
    )


def card(x: float, y: float, w: float, h: float) -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" '
            f'fill="{CARD_BG}" stroke="{CARD_EDGE}" stroke-width="1.5"/>')


def edge_arrow(x1, y1, x2, y2, *, shrink=30.0, color=NODE_EDGE, width=2.4) -> str:
    """Arrow between two node centers, shrunk so it starts/ends at the circles."""
    dx, dy = x2 - x1, y2 - y1
    L = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / L, dy / L
    ax, ay = x1 + ux * shrink, y1 + uy * shrink
    bx, by = x2 - ux * (shrink + 6), y2 - uy * (shrink + 6)
    return (f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" '
            f'stroke="{color}" stroke-width="{width}" marker-end="url(#arrow)"/>')


def build_svg() -> str:
    W, H = 1280, 460
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        '<defs>'
        f'<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{NODE_EDGE}"/></marker>'
        f'<marker id="bigarrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" '
        f'markerHeight="5" orient="auto">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{MUTED}"/></marker>'
        '</defs>',
        f'<rect width="{W}" height="{H}" rx="18" fill="#ffffff" stroke="{CARD_EDGE}"/>',
    ]

    # ======================= panel 1: the model =============================
    px, py, pw, ph = 28, 28, 300, 404
    parts.append(card(px, py, pw, ph))
    parts.append(text("LINEAR GAUSSIAN DAG", x=px + pw / 2, y=py + 32, size=15,
                      fill=MUTED, weight="bold"))
    cx = px + pw / 2
    pos = {0: (cx, py + 88), 1: (cx - 88, py + 196), 2: (cx + 88, py + 196),
           3: (cx, py + 304)}
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        parts.append(edge_arrow(*pos[a], *pos[b]))
    for j, (x, y) in pos.items():
        parts.append(f'<circle cx="{x}" cy="{y}" r="26" fill="{NODE_FILL}" '
                     f'stroke="{NODE_EDGE}" stroke-width="2.5"/>')
        parts.append(sub("V", str(j), x=x, y=y + 6, size=19))
    # edge / noise annotations
    parts.append(sub("A", "ji", x=cx - 78, y=py + 134, size=16, fill=NODE_EDGE))
    parts.append(sub("Σ", "j", x=pos[1][0] - 44, y=pos[1][1] - 30, size=16,
                     fill=MUTED, anchor="end", style="normal"))
    parts.append(text("Vⱼ = Σᵢ Aⱼᵢ Vᵢ + Zⱼ",
                      x=cx, y=py + 366, size=17, fill=INK, style="italic"))
    parts.append(text("local conditional parameters {Aⱼᵢ, Σⱼ}",
                      x=cx, y=py + 390, size=13.5, fill=MUTED))

    # ======================= arrow 1: K-recursion ===========================
    ax0, ax1, ay = 348, 470, 230
    parts.append(f'<line x1="{ax0}" y1="{ay}" x2="{ax1}" y2="{ay}" stroke="{MUTED}" '
                 f'stroke-width="5" marker-end="url(#bigarrow)"/>')
    parts.append(text("K-recursion", x=(ax0 + ax1) / 2, y=ay - 44, size=18,
                      fill=INK, weight="bold"))
    parts.append(text("one forward sweep", x=(ax0 + ax1) / 2, y=ay - 22, size=12.5,
                      fill=MUTED))
    parts.append(f'<rect x="{(ax0 + ax1) / 2 - 56}" y="{ay + 18}" width="112" height="26" '
                 f'rx="13" fill="#fef3c7" stroke="{ACCENT}"/>')
    parts.append(text("∇ autograd", x=(ax0 + ax1) / 2, y=ay + 36, size=13.5,
                      fill=ACCENT, weight="bold"))

    # ======================= panel 2: covariance heatmap ====================
    qx, qy, qw, qh = 490, 28, 358, 404
    parts.append(card(qx, qy, qw, qh))
    parts.append(text("ALL-PAIRS COVARIANCE", x=qx + qw / 2, y=qy + 32, size=15,
                      fill=MUTED, weight="bold"))

    Kf = gbn.k_full(MODEL)                       # real K-recursion output (8x8)
    vals = Kf.numpy() if hasattr(Kf, "numpy") else Kf
    vmax = float(Kf.abs().max())
    n = Kf.shape[0]
    cell = 33
    hx = qx + (qw - n * cell) / 2 + 8
    hy = qy + 66
    for r in range(n):
        for c in range(n):
            parts.append(f'<rect x="{hx + c * cell}" y="{hy + r * cell}" width="{cell - 1}" '
                         f'height="{cell - 1}" rx="3" '
                         f'fill="{cell_color(float(Kf[r, c]), vmax)}"/>')
    # block separators (2-dim nodes -> every 2 cells) + node labels
    for b in range(1, 4):
        d = b * 2 * cell
        parts.append(f'<line x1="{hx + d - 0.5}" y1="{hy}" x2="{hx + d - 0.5}" '
                     f'y2="{hy + n * cell - 1}" stroke="#ffffff" stroke-width="3"/>')
        parts.append(f'<line x1="{hx}" y1="{hy + d - 0.5}" x2="{hx + n * cell - 1}" '
                     f'y2="{hy + d - 0.5}" stroke="#ffffff" stroke-width="3"/>')
    for j in range(4):
        c = hx + (2 * j + 1) * cell
        parts.append(sub("V", str(j), x=c, y=hy - 8, size=13, fill=MUTED))
        parts.append(sub("V", str(j), x=hx - 14, y=hy + (2 * j + 1) * cell + 5,
                         size=13, fill=MUTED))
    parts.append(text("Kⱼₖ = E[Vⱼ Vₖᴴ]",
                      x=qx + qw / 2, y=hy + n * cell + 36, size=17, fill=INK,
                      style="italic"))
    parts.append(text("every node-pair covariance block, in one pass",
                      x=qx + qw / 2, y=hy + n * cell + 60, size=13.5, fill=MUTED))

    # ======================= arrow 2 ========================================
    bx0, bx1 = 868, 916
    parts.append(f'<line x1="{bx0}" y1="{ay}" x2="{bx1}" y2="{ay}" stroke="{MUTED}" '
                 f'stroke-width="5" marker-end="url(#bigarrow)"/>')

    # ======================= panel 3: applications ==========================
    rx, ry, rw, rh = 932, 28, 320, 404
    parts.append(card(rx, ry, rw, rh))
    parts.append(text("ONE ENGINE, EVERYTHING ON TOP", x=rx + rw / 2, y=ry + 32,
                      size=15, fill=MUTED, weight="bold"))
    rows = [
        ("inference", "I(A; B | C) · conditioning · sampling"),
        ("estimation", "MLE · EM · gradient training"),
        ("identifiability", "Fisher metric · Cramér–Rao bounds"),
        ("causality", "do(·) interventions · counterfactuals"),
        ("design", "D-/E-optimal sensor placement"),
    ]
    ry0 = ry + 70
    for i, (head, desc) in enumerate(rows):
        yy = ry0 + i * 66
        parts.append(f'<rect x="{rx + 18}" y="{yy - 21}" width="{rw - 36}" height="54" '
                     f'rx="10" fill="#ffffff" stroke="{CARD_EDGE}"/>')
        parts.append(f'<circle cx="{rx + 38}" cy="{yy + 6}" r="5" fill="{NODE_EDGE}"/>')
        parts.append(text(head, x=rx + 54, y=yy + 1, size=14.5, fill=INK,
                          anchor="start", weight="bold"))
        parts.append(text(desc, x=rx + 54, y=yy + 21, size=12.8, fill=MUTED,
                          anchor="start"))

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    out = pathlib.Path(__file__).with_name("visual_abstract.svg")
    out.write_text(build_svg(), encoding="utf-8")
    print(f"wrote {out}")

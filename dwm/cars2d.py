"""Top-down 2D street rendering with real car sprites, for the world-model GIFs.

The camera follows the AV along its lane; the stop line is drawn as a real intersection (cross street, crosswalk, signal).
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle

ASPHALT, MARK, CURB, WALK, GRASS, BUILD = "#2b2f36", "#d0d7de", "#6e7681", "#3a3f47", "#1f2a20", "#21262d"
LIGHT = {0: "#ff4d4d", 1: "#ffb020", 2: "#3ddc84"}
LANE_W = 3.6


def car(ax, x, y, color, length=4.6, width=1.9, alpha=1.0, outline=None, z=5):
    """A top-down car, rear bumper at x, centred on lane line y, facing +x."""
    L, W = length, width
    body = FancyBboxPatch((x, y - W / 2), L, W, boxstyle="round,pad=0,rounding_size=0.55", fc=color, ec=outline or "none",
                          lw=1.6 if outline else 0, alpha=alpha, zorder=z)
    ax.add_patch(body)
    a = alpha
    for wx in (x + 0.75, x + L - 1.35):  # wheels peeking out at the sides
        for wy in (y - W / 2 - 0.08, y + W / 2 - 0.22):
            ax.add_patch(Rectangle((wx, wy), 0.6, 0.3, fc="#0b0d10", alpha=a, zorder=z - 1))
    # windshield (front), rear window, roof
    ax.add_patch(Polygon([(x + L * 0.62, y - W * 0.38), (x + L * 0.78, y - W * 0.42), (x + L * 0.78, y + W * 0.42),
                          (x + L * 0.62, y + W * 0.38)], fc="#0d1b2a", alpha=0.85 * a, zorder=z + 1))
    ax.add_patch(Polygon([(x + L * 0.14, y - W * 0.40), (x + L * 0.25, y - W * 0.35), (x + L * 0.25, y + W * 0.35),
                          (x + L * 0.14, y + W * 0.40)], fc="#0d1b2a", alpha=0.8 * a, zorder=z + 1))
    ax.add_patch(Rectangle((x + L * 0.27, y - W * 0.33), L * 0.33, W * 0.66, fc="white", alpha=0.10 * a, zorder=z + 1))
    for sy in (y - W * 0.33, y + W * 0.33):
        ax.add_patch(Circle((x + L - 0.12, sy), 0.16, fc="#fff3b0", alpha=a, zorder=z + 2))  # headlights
        ax.add_patch(Rectangle((x + 0.02, sy - 0.12), 0.12, 0.24, fc="#ff3b3b", alpha=a, zorder=z + 2))  # tail lights


def street(ax, x0, x1, line_x, light):
    """Our lane (y = -LANE_W/2, driving +x), the oncoming lane, sidewalks, buildings and an intersection at line_x."""
    ax.add_patch(Rectangle((x0, -LANE_W - 6), x1 - x0, 6, fc=GRASS, zorder=0))
    ax.add_patch(Rectangle((x0, LANE_W), x1 - x0, 6, fc=GRASS, zorder=0))
    rng = np.random.default_rng(3)
    for side in (-1, 1):  # buildings along both sides (fixed to the world, so they scroll past)
        bx = np.floor(x0 / 14) * 14
        while bx < x1:
            w = 9 + 4 * rng.random()
            if not (line_x - 2 < bx + w and bx < line_x + 14):
                y = LANE_W + 2.2 if side > 0 else -LANE_W - 2.2 - 3.6
                ax.add_patch(Rectangle((bx, y), w, 3.6, fc=BUILD, ec="#30363d", lw=0.8, zorder=1))
            bx += 14
    ax.add_patch(Rectangle((x0, -LANE_W - 1.6), x1 - x0, 1.6, fc=WALK, zorder=1))
    ax.add_patch(Rectangle((x0, LANE_W), x1 - x0, 1.6, fc=WALK, zorder=1))
    ax.add_patch(Rectangle((x0, -LANE_W), x1 - x0, 2 * LANE_W, fc=ASPHALT, zorder=1))
    xs = np.arange(np.floor(x0 / 6) * 6, x1, 6.0)
    for x in xs:  # dashed centre line
        if not (line_x < x < line_x + 12):
            ax.plot([x, x + 3], [0, 0], color="#e3b341", lw=1.4, zorder=2)
    for y in (-LANE_W, LANE_W):
        ax.plot([x0, x1], [y, y], color=CURB, lw=1.2, zorder=2)
    if x0 - 15 < line_x < x1:
        ax.add_patch(Rectangle((line_x + 0.5, -LANE_W - 8), 11, 2 * LANE_W + 16, fc=ASPHALT, zorder=1.5))  # cross street
        for k in range(7):  # crosswalk before the intersection
            ax.add_patch(Rectangle((line_x - 3.2, -LANE_W + 0.3 + k * 1.0), 2.6, 0.55, fc=MARK, alpha=0.8, zorder=2))
        ax.plot([line_x, line_x], [-LANE_W, 0], color="white", lw=3, zorder=3)  # stop line
        ax.add_patch(Rectangle((line_x - 0.6, -LANE_W - 3.2), 1.2, 2.6, fc="#161b22", ec="#484f58", lw=0.8, zorder=6))
        ax.add_patch(Circle((line_x, -LANE_W - 1.9), 0.42, fc=LIGHT[light], zorder=7))
        ax.add_patch(Circle((line_x, -LANE_W - 1.9), 1.1, fc=LIGHT[light], alpha=0.18, zorder=6))

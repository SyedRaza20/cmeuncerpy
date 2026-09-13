"""
Interplanetary medium probe-cone figure.

Generates a static, minimal 3D figure of a "cone" of synthetic probes
spanning 0.33 AU to 1.0 AU radially, with Earth at the center of the
cone's far face (1 AU) and the Sun at the origin. Styled for use in an
academic presentation (dark background, flat dots, no shading/animation).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

# ---------------------------------------------------------------------------
# Geometry parameters
# ---------------------------------------------------------------------------
r_min, r_max = 0.33, 1.0     # radial extent [AU]
n_r = 20                     # number of radial "surfaces" (shells)

n_ang = 31                   # number of samples per angular axis
ang_min, ang_max = -30, 30    # longitude/latitude extent [deg], 2-deg steps

r_vals = np.linspace(r_min, r_max, n_r)
lon_vals = np.linspace(ang_min, ang_max, n_ang)
lat_vals = np.linspace(ang_min, ang_max, n_ang)

# Build the full 3D grid of probe positions (spherical -> Cartesian, AU)
LON, LAT, R = np.meshgrid(
    np.radians(lon_vals), np.radians(lat_vals), r_vals, indexing='ij'
)
X = R * np.cos(LAT) * np.cos(LON)
Y = R * np.cos(LAT) * np.sin(LON)
Z = R * np.sin(LAT)

Xf, Yf, Zf = X.flatten(), Y.flatten(), Z.flatten()

# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
os.makedirs('output', exist_ok=True)

BG_COLOR = '#1c202b'
DOT_COLOR = '#d7dae2'
LINE_COLOR = '#8a90a5'
SUN_COLOR = '#eec24a'
SUN_EDGE = '#f6da85'
EARTH_COLOR = '#4caf78'
EARTH_EDGE = '#8fe0ab'
TEXT_MUTED = '#9aa0b3'

fig = plt.figure(figsize=(10, 8), facecolor=BG_COLOR)
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor(BG_COLOR)
fig.patch.set_facecolor(BG_COLOR)

# Probe dots
ax.scatter(
    Xf, Yf, Zf,
    s=1.1, c=DOT_COLOR, alpha=0.35,
    depthshade=False, linewidths=0, zorder=3
)

# ---------------------------------------------------------------------------
# Cone outline: near face, far face, and four corner edge-rays
# ---------------------------------------------------------------------------
boundary_lon = np.concatenate([
    np.linspace(ang_min, ang_max, 60), np.full(60, ang_max),
    np.linspace(ang_max, ang_min, 60), np.full(60, ang_min)
])
boundary_lat = np.concatenate([
    np.full(60, ang_min), np.linspace(ang_min, ang_max, 60),
    np.full(60, ang_max), np.linspace(ang_max, ang_min, 60)
])
blon, blat = np.radians(boundary_lon), np.radians(boundary_lat)

# Far face (1 AU) and near face (0.33 AU) rims
for rr, alpha, lw in [(r_max, 0.9, 1.3), (r_min, 0.55, 1.0)]:
    bx = rr * np.cos(blat) * np.cos(blon)
    by = rr * np.cos(blat) * np.sin(blon)
    bz = rr * np.sin(blat)
    ax.plot(bx, by, bz, color=LINE_COLOR, lw=lw, alpha=alpha, zorder=4)

# Four corner edge-rays connecting near face to far face
for lo in [ang_min, ang_max]:
    for la in [ang_min, ang_max]:
        rlo, rla = np.radians(lo), np.radians(la)
        rs = np.array([r_min, r_max])
        cx = rs * np.cos(rla) * np.cos(rlo)
        cy = rs * np.cos(rla) * np.sin(rlo)
        cz = rs * np.sin(rla)
        ax.plot(cx, cy, cz, color=LINE_COLOR, lw=1.0, alpha=0.75, zorder=4)

# Sun-Earth axis (dashed guide line)
ax.plot([0, 1.02], [0, 0], [0, 0], color='#5b5f6e', lw=0.8, ls='--',
        alpha=0.6, zorder=1)

# ---------------------------------------------------------------------------
# Sun, Earth, and 0.33 markers
# ---------------------------------------------------------------------------
ax.scatter([0], [0], [0], s=380, c=SUN_COLOR, edgecolors=SUN_EDGE,
           linewidths=1.2, zorder=10)
ax.text(-0.02, -0.06, -0.13, 'Sun', color=SUN_COLOR, fontsize=13,
        ha='center', weight='bold')

ax.scatter([1.0], [0], [0], s=110, c=EARTH_COLOR, edgecolors=EARTH_EDGE,
           linewidths=1.0, zorder=10)
ax.text(1.0, 0.10, 0.10, 'Earth (1 AU)', color='#7fd39a', fontsize=12,
        ha='center', weight='bold')

ax.scatter([0.33], [0], [0], s=110, c=EARTH_COLOR, edgecolors=EARTH_EDGE,
           linewidths=1.0, zorder=10)
ax.text(0.33, 0.10, 0.06, '0.33 AU', color='#7fd39a', fontsize=12,
        ha='center', weight="bold")

# ---------------------------------------------------------------------------
# View, limits, and caption
# ---------------------------------------------------------------------------
ax.set_box_aspect([1.3, 0.75, 0.55])
ax.view_init(elev=14, azim=-62)
ax.set_axis_off()
ax.set_xlim(-0.15, 1.1)
ax.set_ylim(-0.55, 0.55)
ax.set_zlim(-0.4, 0.4)

fig.text(
    0.5, 0.03,
    f"Synthetic probe grid: {n_r} radial surfaces ({r_min}\u2013{r_max} AU) "
    f"\u00d7 {n_ang} longitude \u00d7 {n_ang} latitude samples "
    f"({ang_min}\u00b0 to {ang_max}\u00b0, 2\u00b0 steps)",
    color=TEXT_MUTED, fontsize=10.5, ha='center', style='italic'
)

plt.tight_layout()
plt.savefig('figures/interplanetary_earth_probe_cone.png', dpi=320,
            facecolor=BG_COLOR, bbox_inches='tight')
plt.close()

print(f"Saved figure with {n_r * n_ang * n_ang} total probes.")
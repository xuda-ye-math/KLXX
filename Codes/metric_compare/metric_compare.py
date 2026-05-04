import csv
import math
import os

import torch
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

SIGMAS   = [2.0 ** (-k) for k in range(8)]
X_MIN    = -12.0
X_MAX    =  12.0
N_GRID   = 8001
MARKER_SIZE = 3.0
LINE_WIDTH  = 1.0
HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_CSV  = os.path.join(HERE, "metrics.csv")
OUT_PLOT = os.path.join(HERE, "metrics.png")

SQRT_2PI = math.sqrt(2.0 * math.pi)

x = torch.linspace(X_MIN, X_MAX, N_GRID, dtype=torch.float64)

V  = 0.5 * x * x
nu = torch.exp(-V) / SQRT_2PI

rows = []
for sigma in SIGMAS:
    phi   = torch.exp(-0.5 * (x / sigma) ** 2) / SQRT_2PI
    U     = 0.5 * x * x + phi
    e_mU  = torch.exp(-U)
    Z     = torch.trapezoid(e_mU, x)
    mu    = e_mU / Z

    s     = (x / sigma ** 2) * phi

    kl_mn = torch.trapezoid(mu * torch.log(mu / nu), x)
    kl_nm = torch.trapezoid(nu * torch.log(nu / mu), x)
    tv    = 0.5 * torch.trapezoid(torch.abs(mu - nu), x)
    f_mn  = torch.trapezoid(mu * s * s, x)
    f_nm  = torch.trapezoid(nu * s * s, x)
    db_mn = torch.trapezoid(nu * torch.abs(s), x)
    db_nm = torch.trapezoid(mu * torch.abs(s), x)

    rows.append([sigma, Z.item(),
                 kl_mn.item(), kl_nm.item(), tv.item(),
                 f_mn.item(), f_nm.item(), db_mn.item(), db_nm.item()])

header = ["sigma", "Z", "KL_mn", "KL_nm", "TV",
          "Fisher_mn", "Fisher_nm", "DB_mn", "DB_nm"]

print(f"{'sigma':>10} {'Z':>10} {'KL_mn':>12} {'KL_nm':>12} {'TV':>12} "
      f"{'Fisher_mn':>14} {'Fisher_nm':>14} {'DB_mn':>12} {'DB_nm':>12}")
for r in rows:
    print(f"{r[0]:>10.5g} {r[1]:>10.6f} {r[2]:>12.4e} {r[3]:>12.4e} {r[4]:>12.4e} "
          f"{r[5]:>14.4e} {r[6]:>14.4e} {r[7]:>12.4e} {r[8]:>12.4e}")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for r in rows:
        writer.writerow(r)

sigmas_arr = [r[0] for r in rows]
series = [
    (r"$\mathrm{KL}(\mu_\sigma\|\nu)$",     [r[2] for r in rows], "C0", "-"),
    (r"$\mathrm{KL}(\nu\|\mu_\sigma)$",     [r[3] for r in rows], "C0", "--"),
    (r"$\mathrm{TV}(\mu_\sigma,\nu)$",      [r[4] for r in rows], "C1", "-"),
    (r"$\mathrm{Fisher}(\mu_\sigma\|\nu)$", [r[5] for r in rows], "C2", "-"),
    (r"$\mathrm{Fisher}(\nu\|\mu_\sigma)$", [r[6] for r in rows], "C2", "--"),
    (r"$\mathrm{DB}(\mu_\sigma\|\nu)$",     [r[7] for r in rows], "C3", "-"),
    (r"$\mathrm{DB}(\nu\|\mu_\sigma)$",     [r[8] for r in rows], "C3", "--"),
]

plt.figure(figsize=(4.5, 3))
for label, ys, color, ls in series:
    plt.plot(sigmas_arr, ys, marker="o", color=color, linestyle=ls,
             markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=label)
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xlabel(r"$\sigma$")
plt.ylabel("metric value")
plt.title(r"Metrics between $\mu_\sigma$ and $\nu$")
plt.gca().invert_xaxis()
plt.grid(True, which="both", alpha=0.3)
plt.legend(fontsize=8.5, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=300)
print(f"\nWrote {OUT_CSV}\nWrote {OUT_PLOT}")

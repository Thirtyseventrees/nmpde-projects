import csv
import matplotlib.pyplot as plt
import sys
import os

args = sys.argv[1:]
show_plot = False
if "--show" in args:
    show_plot = True
    args.remove("--show")

csv_file = args[0] if len(args) > 0 else "energy-cd-lumped-p1.csv"

time_vals = []
energy_vals = []

with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        time_vals.append(float(row["time"]))
        energy_vals.append(float(row["energy"]))

E0 = energy_vals[0]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Absolute energy.
axes[0].plot(time_vals, energy_vals, linewidth=1.2, color="tab:blue")
axes[0].set_xlabel("Time $t$", fontsize=13)
axes[0].set_ylabel("Discrete energy $E(t)$", fontsize=13)
axes[0].set_title("Energy vs time", fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].ticklabel_format(useOffset=False)

# Relative energy drift.
rel_drift = [(e - E0) / E0 * 100.0 for e in energy_vals]
axes[1].plot(time_vals, rel_drift, linewidth=1.2, color="tab:red")
axes[1].set_xlabel("Time $t$", fontsize=13)
axes[1].set_ylabel("$(E(t) - E_0)/E_0$ [%]", fontsize=13)
axes[1].set_title("Relative energy drift", fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].axhline(0, color="black", linewidth=0.5)

fig.suptitle(
    f"Energy diagnostics ({os.path.basename(csv_file)})",
    fontsize=14,
    y=1.02,
)
fig.tight_layout()

out_path = os.path.splitext(csv_file)[0] + ".png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
if show_plot:
    plt.show()

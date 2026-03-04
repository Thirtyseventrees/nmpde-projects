import csv
import matplotlib.pyplot as plt
import sys
import os

args = sys.argv[1:]
show_plot = False
if "--show" in args:
    show_plot = True
    args.remove("--show")

csv_file = args[0] if len(args) > 0 else "error-cd-lumped-p1.csv"

time_vals = []
l2_vals = []
h1_vals = []

with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        time_vals.append(float(row["time"]))
        l2_vals.append(float(row["L2_error"]))
        if "H1_error" in row and row["H1_error"] != "":
            h1_vals.append(float(row["H1_error"]))

if not l2_vals:
    raise RuntimeError(f"No data rows found in {csv_file}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Error (linear scale).
axes[0].plot(time_vals, l2_vals, linewidth=1.2, color="tab:green", label="L2")
if h1_vals and len(h1_vals) == len(time_vals):
    axes[0].plot(time_vals, h1_vals, linewidth=1.2, color="tab:purple", label="H1")
axes[0].set_xlabel("Time $t$", fontsize=13)
axes[0].set_ylabel("Error", fontsize=13)
axes[0].set_title("Error vs time", fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].ticklabel_format(useOffset=False)
axes[0].legend()

# Error (log scale).
axes[1].semilogy(time_vals, l2_vals, linewidth=1.2, color="tab:orange", label="L2")
if h1_vals and len(h1_vals) == len(time_vals):
    axes[1].semilogy(time_vals, h1_vals, linewidth=1.2, color="tab:red", label="H1")
axes[1].set_xlabel("Time $t$", fontsize=13)
axes[1].set_ylabel("Error (log scale)", fontsize=13)
axes[1].set_title("Error (semilog-y)", fontsize=14)
axes[1].grid(True, which="both", alpha=0.3)
axes[1].legend()

fig.suptitle(
    f"Error diagnostics ({os.path.basename(csv_file)})",
    fontsize=14,
    y=1.02,
)
fig.tight_layout()

out_path = os.path.splitext(csv_file)[0] + ".png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
if show_plot:
    plt.show()

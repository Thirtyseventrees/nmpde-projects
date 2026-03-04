import matplotlib.pyplot as plt
import meshio
import sys
import os
import re


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_3d_surface.py [--dt <value>] file1.vtu [file2.vtu ...]")
        sys.exit(1)

    dt = None
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == "--dt":
        dt = float(args[1])
        args = args[2:]

    if not args:
        print("No VTU files provided.")
        sys.exit(1)

    vtu_files = args
    n = len(vtu_files)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    for idx, vtu_file in enumerate(vtu_files):
        mesh = meshio.read(vtu_file)
        x = mesh.points[:, 0]
        y = mesh.points[:, 1]
        u = mesh.point_data["u"]

        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        surf = ax.plot_trisurf(x, y, u, cmap="coolwarm", edgecolor="none",
                               antialiased=True, alpha=0.95)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")

        basename = os.path.basename(vtu_file)
        match = re.search(r"-(\d+)\.vtu$", basename)
        if match and dt is not None:
            step = int(match.group(1))
            ax.set_title(f"step={step}, t={step * dt:.5f}", fontsize=13)
        elif match:
            step = int(match.group(1))
            ax.set_title(f"step={step}", fontsize=13)
        else:
            ax.set_title(basename)

        ax.view_init(elev=25, azim=-60)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)

    fig.suptitle("Wave equation solution (3D surface)", fontsize=15, y=1.02)
    fig.tight_layout()

    out_path = "surface_3d.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Convergence study plots (h-refinement and dt-refinement).

Scans all result directories to extract final L2/H1 errors, then produces
log-log convergence plots with fitted convergence rates.

Usage:
    python3 plot_convergence.py <result_dir> [--show] [--scheme <cd|newmark>]
                                            [--dt-h <h_value>]
                                            [--dt-values v1,v2,...]
                                            [--dt-mass <lumped|consistent>]
                                            [--dt-p <int>]
"""

import csv
import glob
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------------ helpers

def parse_run_dir_name(name):
    """
    Extract configuration from directory name like:
      mesh-mesh_square_h0.05-mode-eigenmode-time-cd-mass-lumped-p1-dt-0.005-T-2-...
    Returns dict with keys: h, dt, scheme, mass, p  (strings / floats).
    """
    info = {}
    # h
    m = re.search(r'h([\d.]+)', name)
    if m:
        info['h'] = float(m.group(1))
    # dt
    m = re.search(r'-dt-([^-]+)', name)
    if m:
        info['dt'] = float(m.group(1))
    # time scheme
    m = re.search(r'-time-(\w+)', name)
    if m:
        info['scheme'] = m.group(1)
    # mass
    m = re.search(r'-mass-(\w+)', name)
    if m:
        info['mass'] = m.group(1)
    # p (fe degree)
    m = re.search(r'-p(\d+)-', name)
    if m:
        info['p'] = int(m.group(1))
    return info


def read_final_errors(csv_path):
    """Read last row of error CSV and return (L2, H1)."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        return None, None
    last = rows[-1]
    l2 = float(last['L2_error'])
    h1 = float(last.get('H1_error', 0))
    if not math.isfinite(l2) or not math.isfinite(h1):
        return None, None
    return l2, h1


def fit_slope(x, y):
    """Fit slope in log-log space."""
    lx = np.log(np.array(x))
    ly = np.log(np.array(y))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, _ = np.linalg.lstsq(A, ly, rcond=None)[0]
    return slope


def format_dt_tick(x):
    """Compact decimal formatting for dt tick labels."""
    s = f"{x:.6f}".rstrip('0').rstrip('.')
    return s if s else "0"


def close_to_any(value, targets, tol=1e-12):
    if not targets:
        return True
    for t in targets:
        if abs(value - t) <= max(tol, 1e-9 * max(1.0, abs(t))):
            return True
    return False

# ------------------------------------------------------------------ main

def main():
    raw_args = sys.argv[1:]
    show_plot = '--show' in sys.argv
    scheme_filter = None
    dt_h_filter = None
    dt_values_filter = None
    dt_mass_filter = None
    dt_p_filter = None
    args = []
    i = 0
    while i < len(raw_args):
        a = raw_args[i]
        if a == '--show':
            i += 1
            continue
        if a == '--scheme':
            if i + 1 >= len(raw_args):
                print("Usage: python3 plot_convergence.py <result_dir> [--show] [--scheme <cd|newmark>] [--dt-h <h_value>] [--dt-values v1,v2,...]")
                sys.exit(1)
            scheme_filter = raw_args[i + 1]
            i += 2
            continue
        if a == '--dt-h':
            if i + 1 >= len(raw_args):
                print("Usage: python3 plot_convergence.py <result_dir> [--show] [--scheme <cd|newmark>] [--dt-h <h_value>] [--dt-values v1,v2,...]")
                sys.exit(1)
            dt_h_filter = float(raw_args[i + 1])
            i += 2
            continue
        if a == '--dt-values':
            if i + 1 >= len(raw_args):
                print("Usage: python3 plot_convergence.py <result_dir> [--show] [--scheme <cd|newmark>] [--dt-h <h_value>] [--dt-values v1,v2,...] [--dt-mass <lumped|consistent>] [--dt-p <int>]")
                sys.exit(1)
            dt_values_filter = [float(x) for x in raw_args[i + 1].split(',') if x.strip()]
            i += 2
            continue
        if a == '--dt-mass':
            if i + 1 >= len(raw_args):
                print("Usage: python3 plot_convergence.py <result_dir> [--show] [--scheme <cd|newmark>] [--dt-h <h_value>] [--dt-values v1,v2,...] [--dt-mass <lumped|consistent>] [--dt-p <int>]")
                sys.exit(1)
            dt_mass_filter = raw_args[i + 1]
            i += 2
            continue
        if a == '--dt-p':
            if i + 1 >= len(raw_args):
                print("Usage: python3 plot_convergence.py <result_dir> [--show] [--scheme <cd|newmark>] [--dt-h <h_value>] [--dt-values v1,v2,...] [--dt-mass <lumped|consistent>] [--dt-p <int>]")
                sys.exit(1)
            dt_p_filter = int(raw_args[i + 1])
            i += 2
            continue
        args.append(a)
        i += 1

    if not args:
        print("Usage: python3 plot_convergence.py <result_dir> [--show] [--scheme <cd|newmark>] [--dt-h <h_value>] [--dt-values v1,v2,...]")
        sys.exit(1)

    result_dir = args[0]

    # Collect all result sub-directories
    subdirs = sorted(glob.glob(os.path.join(result_dir, '*')))
    subdirs = [d for d in subdirs if os.path.isdir(d)]

    # ===================== h-convergence =====================
    # Group by (scheme, mass, p, dt) → dict of h → (L2, H1)
    h_groups = {}
    for d in subdirs:
        info = parse_run_dir_name(os.path.basename(d))
        if not all(k in info for k in ('h', 'dt', 'scheme', 'mass', 'p')):
            continue
        if scheme_filter and info['scheme'] != scheme_filter:
            continue
        # Find error CSV
        csvs = glob.glob(os.path.join(d, 'error-*.csv'))
        if not csvs:
            continue
        l2, h1 = read_final_errors(csvs[0])
        if l2 is None:
            continue
        key = (info['scheme'], info['mass'], info['p'], info['dt'])
        h_groups.setdefault(key, []).append((info['h'], l2, h1))

    # For h-convergence, pick groups where dt is the smallest (most refined in time)
    # so temporal error doesn't dominate
    h_conv_data = {}
    for key, entries in h_groups.items():
        scheme, mass, p, dt = key
        label_key = (scheme, mass, p)
        if label_key not in h_conv_data or dt < h_conv_data[label_key][0]:
            h_conv_data[label_key] = (dt, sorted(entries, key=lambda x: x[0]))

    if h_conv_data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for (scheme, mass, p), (dt, entries) in sorted(h_conv_data.items()):
            if len(entries) < 2:
                continue
            hs = [e[0] for e in entries]
            l2s = [e[1] for e in entries]
            h1s = [e[2] for e in entries]
            label = f"{scheme}-{mass}-P{p} (dt={dt})"
            slope_l2 = fit_slope(hs, l2s)
            slope_h1 = fit_slope(hs, h1s) if all(v > 0 for v in h1s) else 0

            axes[0].loglog(hs, l2s, 'o-', label=label)
            if all(v > 0 for v in h1s):
                axes[1].loglog(hs, h1s, 's-', label=label)

        # Reference slopes
        h_ref = np.array([0.025, 0.2])
        for ax, (expected_p1, expected_p2, title) in zip(
            axes,
            [(2, 3, '$L^2$ error vs $h$'), (1, 2, '$H^1$ error vs $h$')]
        ):
            y_ref = h_ref ** expected_p1 * 0.5
            ax.loglog(h_ref, y_ref, 'k--', alpha=0.4, label=f'$O(h^{expected_p1})$ ref')
            y_ref2 = h_ref ** expected_p2 * 0.3
            ax.loglog(h_ref, y_ref2, 'k:', alpha=0.4, label=f'$O(h^{expected_p2})$ ref')
            ax.set_xlabel('Mesh size $h$', fontsize=13)
            ax.set_ylabel('Error', fontsize=13)
            ax.set_title(title, fontsize=14)
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(fontsize=9)
            ax.invert_xaxis()

        fig.suptitle('Spatial convergence (h-refinement)', fontsize=15, y=1.02)
        fig.tight_layout()
        out_name = 'convergence_h.png' if not scheme_filter else f'convergence_h_{scheme_filter}.png'
        out = os.path.join(result_dir, out_name)
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved {out}")
        if show_plot:
            plt.show()
        plt.close(fig)

    # ===================== dt-convergence =====================
    dt_groups = {}
    for key, entries in h_groups.items():
        scheme, mass, p, dt = key
        # Group by (scheme, mass, p, h) → dict of dt → (L2, H1)
        for h_val, l2, h1 in entries:
            gk = (scheme, mass, p, h_val)
            dt_groups.setdefault(gk, []).append((dt, l2, h1))

    # Pick groups with the widest dt coverage first; if tied, use smaller h.
    # This avoids accidentally shrinking the dt range when only partial runs
    # exist on the finest mesh.
    dt_conv_data = {}
    for (scheme, mass, p, h_val), dts in dt_groups.items():
        if dt_mass_filter is not None and mass != dt_mass_filter:
            continue
        if dt_p_filter is not None and p != dt_p_filter:
            continue
        if dt_h_filter is not None and abs(h_val - dt_h_filter) > 1e-12:
            continue
        dts_filtered = [entry for entry in dts if close_to_any(entry[0], dt_values_filter)]
        if len(dts_filtered) < 2:
            continue
        label_key = (scheme, mass, p)
        cand_count = len(dts_filtered)
        cand_entries = sorted(dts_filtered, key=lambda x: x[0])
        if label_key not in dt_conv_data:
            dt_conv_data[label_key] = (h_val, cand_count, cand_entries)
        else:
            best_h, best_count, _ = dt_conv_data[label_key]
            if cand_count > best_count or (cand_count == best_count and h_val < best_h):
                dt_conv_data[label_key] = (h_val, cand_count, cand_entries)

    if dt_conv_data:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        dt_ticks = set()
        dt_all = []
        for (scheme, mass, p), (h_val, _, entries) in sorted(dt_conv_data.items()):
            if len(entries) < 2:
                continue
            # Plot from large to small dt for readability.
            entries = sorted(entries, key=lambda x: x[0], reverse=True)
            dts_v = [e[0] for e in entries]
            l2s = [e[1] for e in entries]
            h1s = [e[2] for e in entries]
            dt_ticks.update(dts_v)
            dt_all.extend(dts_v)
            label = f"{scheme}-{mass}-P{p} (h={h_val})"
            slope_l2 = fit_slope(dts_v, l2s) if len(dts_v) >= 2 else 0
            slope_h1 = fit_slope(dts_v, h1s) if len(dts_v) >= 2 and all(v > 0 for v in h1s) else 0

            axes[0].loglog(dts_v, l2s, 'o-', label=label)
            if all(v > 0 for v in h1s):
                axes[1].loglog(dts_v, h1s, 's-', label=label)

        if dt_all:
            dt_ref = np.array([min(dt_all), max(dt_all)])
        else:
            dt_ref = np.array([0.002, 0.04])
        for ax, title in zip(axes, ['$L^2$ error vs $\\Delta t$', '$H^1$ error vs $\\Delta t$']):
            y2 = dt_ref ** 2 * 2.0
            ax.loglog(dt_ref, y2, 'k--', alpha=0.4, label='$O(\\Delta t^2)$ ref')
            ax.set_xlabel('$\\Delta t$', fontsize=13)
            ax.set_ylabel('Error', fontsize=13)
            ax.set_title(title, fontsize=14)
            # Show dt from large -> small, with explicit tick values.
            if dt_ticks:
                ticks = sorted(dt_ticks, reverse=True)
                ax.set_xticks(ticks)
                ax.set_xticklabels([format_dt_tick(t) for t in ticks],
                                   rotation=25,
                                   ha='right')
                dt_min = min(ticks)
                dt_max = max(ticks)
                # Keep x-range tied to selected data, avoiding empty left area.
                ax.set_xlim(dt_max * 1.05, dt_min * 0.95)
            ax.grid(True, which='both', alpha=0.3)
            ax.legend(fontsize=9)

        fig.suptitle('Temporal convergence ($\\Delta t$-refinement)', fontsize=15, y=1.02)
        fig.tight_layout()
        out_name = 'convergence_dt.png' if not scheme_filter else f'convergence_dt_{scheme_filter}.png'
        out = os.path.join(result_dir, out_name)
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved {out}")
        if show_plot:
            plt.show()
        plt.close(fig)

    if not h_conv_data and not dt_conv_data:
        print("[info] Not enough data for convergence plots.")


if __name__ == '__main__':
    main()

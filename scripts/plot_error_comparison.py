#!/usr/bin/env python3
"""
Error comparison (dispersion analysis via error growth).

Overlays L2 / H1 error time-histories from multiple runs to show how
numerical dispersion causes phase error accumulation over time.

Usage:
    python3 plot_error_comparison.py <result_dir> [--show]
"""

import csv
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#2196F3', '#E91E63', '#4CAF50', '#FF9800', '#9C27B0', '#00BCD4',
          '#795548', '#607D8B']
STYLES = ['-', '--', '-.', ':']


def parse_run_dir_name(name):
    info = {}
    m = re.search(r'h([\d.]+)', name)
    if m: info['h'] = float(m.group(1))
    m = re.search(r'-dt-([^-]+)', name)
    if m: info['dt'] = float(m.group(1))
    m = re.search(r'-time-(\w+)', name)
    if m: info['scheme'] = m.group(1)
    m = re.search(r'-mass-(\w+)', name)
    if m: info['mass'] = m.group(1)
    m = re.search(r'-p(\d+)-', name)
    if m: info['p'] = int(m.group(1))
    m = re.search(r'-bc-(\w+)', name)
    if m: info['bc'] = m.group(1)
    m = re.search(r'-nmb-([^-]+)', name)
    if m: info['beta'] = float(m.group(1))
    m = re.search(r'-nmg-([^-]+)', name)
    if m: info['gamma'] = float(m.group(1))
    m = re.search(r'-errstep-(\d+)', name)
    if m: info['errstep'] = int(m.group(1))
    return info


def read_error_csv(csv_path):
    times, l2s, h1s = [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = float(row['time'])
            l2 = float(row['L2_error'])
            h1v = row.get('H1_error', '')
            h1 = float(h1v) if h1v else 0.0
            if not (np.isfinite(t) and np.isfinite(l2) and np.isfinite(h1)):
                continue
            times.append(t)
            l2s.append(l2)
            h1s.append(h1)
    return np.array(times), np.array(l2s), np.array(h1s)


def make_label(info):
    s = info.get('scheme', '?')
    m = info.get('mass', '?')
    p = info.get('p', 1)
    return f"{s}-{m}-P{p}"


def add_dedup_legend(ax, fontsize=9):
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l not in uniq:
            uniq[l] = h
    if uniq:
        ax.legend(list(uniq.values()), list(uniq.keys()), fontsize=fontsize)


def main():
    args = [a for a in sys.argv[1:] if a != '--show']
    show_plot = '--show' in sys.argv

    if not args:
        print("Usage: python3 plot_error_comparison.py <result_dir> [--show]")
        sys.exit(1)

    result_dir = args[0]
    subdirs = sorted(glob.glob(os.path.join(result_dir, '*')))
    subdirs = [d for d in subdirs if os.path.isdir(d)]

    # Group by (h, dt) for scheme comparison
    groups = {}
    for d in subdirs:
        name = os.path.basename(d)
        info = parse_run_dir_name(name)
        if not all(k in info for k in ('h', 'dt', 'scheme', 'mass')):
            continue
        csvs = glob.glob(os.path.join(d, 'error-*.csv'))
        if not csvs:
            continue
        key = (info['h'], info['dt'])
        groups.setdefault(key, []).append((name, info, csvs[0]))

    # Baseline group for report: h=0.05, dt=0.005, P1, homogeneous BC when present.
    # For Newmark runs, keep only default (beta,gamma)=(0.25,0.5).
    baseline_candidates = []
    key_baseline = (0.05, 0.005)
    if key_baseline in groups:
        for name, info, csv_path in groups[key_baseline]:
            # Accept both old and new directory naming; if bc tag exists, require homogeneous.
            if 'bc' in info and info.get('bc') != 'homogeneous':
                continue
            if info.get('p', 1) != 1:
                continue
            if info.get('scheme') == 'newmark':
                beta = info.get('beta', 0.25)
                gamma = info.get('gamma', 0.5)
                if abs(beta - 0.25) > 1e-12 or abs(gamma - 0.5) > 1e-12:
                    continue
            baseline_candidates.append((name, info, csv_path))

    # Deduplicate labels and prefer runs with errstep=1 (full time history).
    baseline_unique = {}
    for name, info, csv_path in baseline_candidates:
        key = make_label(info)
        current = baseline_unique.get(key)
        score = info.get('errstep', 0)
        if current is None or score > current[0]:
            baseline_unique[key] = (score, info, csv_path)
    baseline_entries = [(v[1], v[2]) for k, v in sorted(baseline_unique.items())]

    if len(baseline_entries) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        for idx, (info, csv_path) in enumerate(
                sorted(baseline_entries, key=lambda x: make_label(x[0]))):
            times, l2s, h1s = read_error_csv(csv_path)
            if len(times) == 0:
                continue
            label = make_label(info)
            color = COLORS[idx % len(COLORS)]
            style = STYLES[idx % len(STYLES)]
            axes[0, 0].plot(times, l2s, color=color, linestyle=style,
                            linewidth=1.5, marker='o', markersize=3, label=label)
            axes[0, 1].semilogy(times, l2s, color=color, linestyle=style,
                                linewidth=1.5, marker='o', markersize=3, label=label)
            if np.any(h1s > 0):
                axes[1, 0].plot(times, h1s, color=color, linestyle=style,
                                linewidth=1.5, marker='o', markersize=3, label=label)
                axes[1, 1].semilogy(times, h1s, color=color, linestyle=style,
                                    linewidth=1.5, marker='o', markersize=3, label=label)

        titles = [
            ('$L^2$ error vs time', 'Error'),
            ('$L^2$ error vs time (log)', 'Error (log)'),
            ('$H^1$ error vs time', 'Error'),
            ('$H^1$ error vs time (log)', 'Error (log)'),
        ]
        for ax, (title, ylabel) in zip(axes.flat, titles):
            ax.set_xlabel('Time $t$', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=13)
            ax.grid(True, which='both', alpha=0.3)
            add_dedup_legend(ax, fontsize=9)
        fig.suptitle('Error comparison — baseline experiment set',
                     fontsize=14, y=1.02)
        fig.tight_layout()
        out = os.path.join(result_dir, 'error_comparison_baseline.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved {out}")
        if show_plot:
            plt.show()
        plt.close(fig)

    comparison_groups = {k: v for k, v in groups.items() if len(v) >= 2}

    if not comparison_groups:
        print("[info] No scheme comparison data found.")
        return

    for (h, dt), entries in sorted(comparison_groups.items()):
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))

        unique = {}
        for name, info, csv_path in entries:
            unique[make_label(info)] = (info, csv_path)
        unique_entries = [unique[k] for k in sorted(unique.keys())]

        for idx, (info, csv_path) in enumerate(unique_entries):
            times, l2s, h1s = read_error_csv(csv_path)
            if len(times) == 0:
                continue
            label = make_label(info)
            color = COLORS[idx % len(COLORS)]
            style = STYLES[idx % len(STYLES)]

            # L2 error (linear)
            axes[0, 0].plot(times, l2s, color=color, linestyle=style,
                            linewidth=1.5, marker='o', markersize=3, label=label)
            # L2 error (log)
            axes[0, 1].semilogy(times, l2s, color=color, linestyle=style,
                                linewidth=1.5, marker='o', markersize=3, label=label)
            # H1 error (linear)
            if np.any(h1s > 0):
                axes[1, 0].plot(times, h1s, color=color, linestyle=style,
                                linewidth=1.5, marker='o', markersize=3, label=label)
                # H1 error (log)
                axes[1, 1].semilogy(times, h1s, color=color, linestyle=style,
                                    linewidth=1.5, marker='o', markersize=3, label=label)

        titles = [
            ('$L^2$ error vs time', 'Error'),
            ('$L^2$ error vs time (log)', 'Error (log)'),
            ('$H^1$ error vs time', 'Error'),
            ('$H^1$ error vs time (log)', 'Error (log)'),
        ]
        for ax, (title, ylabel) in zip(axes.flat, titles):
            ax.set_xlabel('Time $t$', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=13)
            ax.grid(True, which='both', alpha=0.3)
            add_dedup_legend(ax, fontsize=9)
        fig.suptitle(
            f'Error comparison — Numerical dispersion  '
            f'($h={h}$, $\\Delta t={dt}$)',
            fontsize=14, y=1.02)
        fig.tight_layout()

        out = os.path.join(result_dir, f'error_comparison_h{h}_dt{dt}.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"Saved {out}")
        if show_plot:
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()

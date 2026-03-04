#!/usr/bin/env python3
"""
Generate report supplement plots:
1) Newmark parameter comparison (energy dissipation)
2) Homogeneous vs driven boundary energy comparison
3) CFL sweep indicator for explicit central difference
"""

import csv
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_run_dir_name(name):
    info = {}
    m = re.search(r'h([\d.]+)', name)
    if m:
        info['h'] = float(m.group(1))
    m = re.search(r'-dt-([^-]+)', name)
    if m:
        info['dt'] = float(m.group(1))
    m = re.search(r'-time-(\w+)', name)
    if m:
        info['scheme'] = m.group(1)
    m = re.search(r'-mass-(\w+)', name)
    if m:
        info['mass'] = m.group(1)
    m = re.search(r'-p(\d+)-', name)
    if m:
        info['p'] = int(m.group(1))
    m = re.search(r'-bc-(\w+)', name)
    if m:
        info['bc'] = m.group(1)
    m = re.search(r'-nmb-([^-]+)', name)
    if m:
        info['beta'] = float(m.group(1))
    m = re.search(r'-nmg-([^-]+)', name)
    if m:
        info['gamma'] = float(m.group(1))
    return info


def read_energy_csv(csv_path):
    t, e = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tv = float(row['time'])
            ev = float(row['energy'])
            if np.isfinite(tv) and np.isfinite(ev):
                t.append(tv)
                e.append(ev)
    return np.array(t), np.array(e)


def gather_runs(result_dir):
    out = []
    for d in sorted(glob.glob(os.path.join(result_dir, '*'))):
        if not os.path.isdir(d):
            continue
        name = os.path.basename(d)
        info = parse_run_dir_name(os.path.basename(d))
        energy_csvs = glob.glob(os.path.join(d, 'energy-*.csv'))
        if not energy_csvs:
            continue
        out.append((d, name, info, energy_csvs[0]))
    return out


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def plot_newmark_param_comparison(result_dir, runs):
    selected = []
    for _, name, info, csv_path in runs:
        if '-bc-' not in name:
            continue
        if info.get('scheme') != 'newmark':
            continue
        if info.get('mass') != 'consistent':
            continue
        if info.get('bc', 'homogeneous') != 'homogeneous':
            continue
        if info.get('p') != 1:
            continue
        if not close(info.get('h', -1.0), 0.05):
            continue
        if not close(info.get('dt', -1.0), 0.005):
            continue
        if 'beta' in info and 'gamma' in info:
            selected.append((info, csv_path))

    # Keep unique (beta,gamma)
    dedup = {}
    for info, csv_path in selected:
        dedup[(info['beta'], info['gamma'])] = (info, csv_path)
    selected = [dedup[k] for k in sorted(dedup.keys())]

    if len(selected) < 2:
        print('[info] Skip Newmark param plot: not enough runs.')
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for info, csv_path in selected:
        t, e = read_energy_csv(csv_path)
        if len(t) == 0:
            continue
        e0 = e[0] if abs(e[0]) > 1e-16 else 1e-16
        drift = (e - e0) / e0 * 100.0
        ax.plot(t, drift, linewidth=1.6,
                label=f"beta={info['beta']:.2f}, gamma={info['gamma']:.2f}")

    ax.axhline(0.0, color='black', linewidth=0.7)
    ax.set_xlabel('Time t')
    ax.set_ylabel('Relative energy drift [%]')
    ax.set_title('Newmark parameter impact on algorithmic dissipation')
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = os.path.join(result_dir, 'supplement_newmark_dissipation.png')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f'Saved {out}')


def plot_boundary_driving_comparison(result_dir, runs):
    selected = {}
    for _, name, info, csv_path in runs:
        if '-bc-' not in name:
            continue
        if info.get('scheme') != 'newmark':
            continue
        if info.get('mass') != 'consistent':
            continue
        if info.get('p') != 1:
            continue
        if not close(info.get('h', -1.0), 0.05):
            continue
        if not close(info.get('dt', -1.0), 0.005):
            continue
        if not close(info.get('beta', -1.0), 0.25):
            continue
        if not close(info.get('gamma', -1.0), 0.5):
            continue
        bc = info.get('bc', 'homogeneous')
        selected[bc] = (info, csv_path)

    if 'homogeneous' not in selected or 'driven' not in selected:
        print('[info] Skip boundary driving plot: homogeneous/driven pair missing.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for bc, color in [('homogeneous', '#1f77b4'), ('driven', '#d62728')]:
        _, csv_path = selected[bc]
        t, e = read_energy_csv(csv_path)
        if len(t) == 0:
            continue
        e0 = e[0] if abs(e[0]) > 1e-16 else 1e-16
        drift = (e - e0) / e0 * 100.0
        axes[0].plot(t, e, color=color, linewidth=1.6, label=bc)
        axes[1].plot(t, drift, color=color, linewidth=1.6, label=bc)

    axes[0].set_xlabel('Time t')
    axes[0].set_ylabel('Discrete energy E(t)')
    axes[0].set_title('Energy evolution')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].axhline(0.0, color='black', linewidth=0.7)
    axes[1].set_xlabel('Time t')
    axes[1].set_ylabel('Relative energy drift [%]')
    axes[1].set_title('Relative drift vs initial energy')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle('Boundary driving effect on energy', y=1.02)
    fig.tight_layout()

    out = os.path.join(result_dir, 'supplement_boundary_energy.png')
    fig.savefig(out, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def plot_cfl_sweep(result_dir, runs):
    points = []
    # Keep one run per CFL to avoid legacy duplicate directories.
    by_cfl = {}
    for _, name, info, csv_path in runs:
        if '-bc-' not in name:
            continue
        if info.get('scheme') != 'cd':
            continue
        if info.get('mass') != 'lumped':
            continue
        if info.get('p') != 1:
            continue
        if info.get('bc', 'homogeneous') != 'homogeneous':
            continue
        if not close(info.get('h', -1.0), 0.025):
            continue

        dt = info.get('dt')
        h = info.get('h')
        if dt is None or h is None:
            continue

        t, e = read_energy_csv(csv_path)
        if len(e) == 0:
            continue
        e0 = e[0] if abs(e[0]) > 1e-16 else 1e-16
        ratio = np.nanmax(np.abs(e)) / abs(e0)
        final_ratio = abs(e[-1]) / abs(e0)
        stable_flag = int(np.isfinite(ratio) and ratio < 10.0)
        cfl = dt / h
        by_cfl[cfl] = (cfl, ratio, final_ratio, stable_flag)

    points = sorted(by_cfl.values(), key=lambda x: x[0])
    if len(points) < 2:
        print('[info] Skip CFL sweep plot: not enough runs.')
        return

    cfl = np.array([p[0] for p in points])
    ratio = np.array([p[1] for p in points])
    final_ratio = np.array([p[2] for p in points])
    stable = np.array([p[3] for p in points])

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    axes[0].semilogy(cfl, ratio, 'o-', linewidth=1.6, label='max |E| / |E0|')
    axes[0].semilogy(cfl, final_ratio, 's--', linewidth=1.4, label='|E(T)| / |E0|')
    axes[0].set_xlabel('CFL = dt / h')
    axes[0].set_ylabel('Energy growth indicator')
    axes[0].set_title('Explicit central difference stability scan')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].legend()

    axes[1].plot(cfl, stable, 'o-', color='#2ca02c', linewidth=1.6)
    axes[1].set_xlabel('CFL = dt / h')
    axes[1].set_ylabel('Empirical stable flag')
    axes[1].set_yticks([0, 1])
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].set_title('1: stable-like, 0: blow-up-like')
    axes[1].grid(True, alpha=0.3)

    out = os.path.join(result_dir, 'supplement_cfl_scan.png')
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f'Saved {out}')


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 plot_report_supplement.py <result_dir>')
        sys.exit(1)

    result_dir = sys.argv[1]
    runs = gather_runs(result_dir)
    if not runs:
        print('[info] No result runs found.')
        return

    plot_newmark_param_comparison(result_dir, runs)
    plot_boundary_driving_comparison(result_dir, runs)
    plot_cfl_sweep(result_dir, runs)


if __name__ == '__main__':
    main()

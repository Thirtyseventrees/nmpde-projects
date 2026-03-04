#!/usr/bin/env python3
"""
Build dispersion/dissipation diagnostics from C++ simulation results.

This script reads per-run probe signals saved by Wave2D (probe-*.csv)
and estimates:
  - numerical frequency omega_d  (via FFT peak)
  - per-step amplification |G|   (via envelope ratio)

From those estimates, it generates the report figures:
  - dispersion_relation.png
  - dissipation_amplification_central.png
  - dissipation_amplification_newmark.png

Usage:
  python3 scripts/plot_dispersion_from_results.py result/ \
      [--h 0.05] [--p 1]
      [--cfl-values 0.5,0.8,1.0] [--cd-cfl-values 0.1]
      [--amp-cfl-values 0.1] [--semi-cfl 0.1] [--show]
"""

import csv
import glob
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
    m = re.search(r'-mx-(\d+)-my-(\d+)', name)
    if m:
        info['mx'] = int(m.group(1))
        info['my'] = int(m.group(2))
    return info


def read_meta_csv(meta_path):
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    row = rows[0]
    out = {}
    for k in ('h_min', 'dt', 'cfl', 'mode_x', 'mode_y', 'omega'):
        if k in row and row[k] != '':
            try:
                out[k] = float(row[k])
            except ValueError:
                pass
    return out


def read_probe_csv(probe_path):
    t, u1, u2 = [], [], []
    with open(probe_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tv = float(row['time'])
            except Exception:
                continue
            v1 = float(row.get('u_probe1', 'nan'))
            v2 = float(row.get('u_probe2', 'nan'))
            t.append(tv)
            u1.append(v1)
            u2.append(v2)
    return np.array(t), np.array(u1), np.array(u2)


def choose_probe_signal(times, u1, u2):
    m1 = np.isfinite(times) & np.isfinite(u1)
    m2 = np.isfinite(times) & np.isfinite(u2)

    std1 = np.std(u1[m1]) if np.any(m1) else 0.0
    std2 = np.std(u2[m2]) if np.any(m2) else 0.0

    if std1 >= std2 and np.any(m1):
        return times[m1], u1[m1]
    if np.any(m2):
        return times[m2], u2[m2]

    # Last resort: weighted combination with finite masks.
    m = np.isfinite(times) & np.isfinite(u1) & np.isfinite(u2)
    if np.any(m):
        return times[m], 0.618 * u1[m] + 0.382 * u2[m]
    return np.array([]), np.array([])


def estimate_omega_zero_crossings(times, signal):
    if len(times) < 8:
        return math.nan
    y = signal - np.mean(signal)
    if np.max(np.abs(y)) < 1e-14:
        return math.nan

    crossings = []
    for i in range(len(y) - 1):
        y0 = y[i]
        y1 = y[i + 1]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y0 == 0.0:
            crossings.append(times[i])
            continue
        if y0 * y1 < 0.0:
            dt = times[i + 1] - times[i]
            if dt <= 0.0:
                continue
            t_cross = times[i] - y0 * dt / (y1 - y0)
            crossings.append(t_cross)

    if len(crossings) < 3:
        return math.nan

    half_periods = np.diff(np.array(crossings))
    half_periods = half_periods[np.isfinite(half_periods) & (half_periods > 0.0)]
    if len(half_periods) == 0:
        return math.nan
    half_T = float(np.median(half_periods))
    return np.pi / half_T


def estimate_omega_fft(times, signal):
    if len(times) < 16:
        return math.nan
    dt = np.median(np.diff(times))
    if not np.isfinite(dt) or dt <= 0.0:
        return math.nan

    y = signal - np.mean(signal)
    if np.max(np.abs(y)) < 1e-14:
        return math.nan

    win = np.hanning(len(y))
    spec = np.fft.rfft(y * win)
    freq = np.fft.rfftfreq(len(y), d=dt)
    amp = np.abs(spec)
    if len(amp) < 2:
        return math.nan

    amp[0] = 0.0
    idx = int(np.argmax(amp))
    if idx <= 0:
        return math.nan
    return 2.0 * np.pi * freq[idx]


def estimate_omega(times, signal):
    # Prefer zero-crossing estimate (robust to mild growth/decay), then FFT fallback.
    w = estimate_omega_zero_crossings(times, signal)
    if np.isfinite(w) and w > 0.0:
        return w
    return estimate_omega_fft(times, signal)


def estimate_amplification_per_step(signal):
    if len(signal) < 12:
        return math.nan
    y = signal - np.mean(signal)
    a = np.abs(y)
    if np.max(a) < 1e-14:
        return math.nan

    n = len(a)
    w = max(4, n // 10)
    # Use RMS envelope in early/late windows (more phase-robust than max).
    a0 = float(np.sqrt(np.mean(y[:w] * y[:w])))
    a1 = float(np.sqrt(np.mean(y[-w:] * y[-w:])))
    if a0 <= 1e-14:
        return math.nan

    nstep = max(1, n - w)
    return (a1 / a0) ** (1.0 / nstep)


def nearest_target(x, targets, tol):
    if not targets:
        return None
    best = min(targets, key=lambda t: abs(x - t))
    if abs(x - best) <= tol:
        return best
    return None


def dedup_by_kh(records):
    """Keep one record per kh (nearest CFL already selected outside)."""
    out = {}
    for r in records:
        key = round(r['kh'], 6)
        cur = out.get(key)
        if cur is None:
            out[key] = r
            continue
        # Prefer record with finite and closer-to-1 amplification.
        g_new = r.get('G', math.nan)
        g_old = cur.get('G', math.nan)
        score_new = abs(g_new - 1.0) if np.isfinite(g_new) else 1e9
        score_old = abs(g_old - 1.0) if np.isfinite(g_old) else 1e9
        if score_new < score_old:
            out[key] = r
    return sorted(out.values(), key=lambda x: x['kh'])


def set_dynamic_ylim(ax, curves, include_values=None, pad_ratio=0.08):
    vals = []
    for c in curves:
        arr = np.asarray(c, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(arr)
    if include_values:
        vals.append(np.asarray(include_values, dtype=float))
    if not vals:
        return
    all_vals = np.concatenate(vals)
    y_min = float(np.min(all_vals))
    y_max = float(np.max(all_vals))
    if y_max <= y_min:
        c = y_min
        half = max(1e-3, 0.05 * max(1.0, abs(c)))
        ax.set_ylim(c - half, c + half)
        return
    span = y_max - y_min
    pad = max(1e-4, pad_ratio * span)
    ax.set_ylim(y_min - pad, y_max + pad)


def collect_records(result_dir, h_filter, p_filter, min_kh=1.0):
    subdirs = sorted(glob.glob(os.path.join(result_dir, '*')))
    subdirs = [d for d in subdirs if os.path.isdir(d)]

    records = []
    for d in subdirs:
        name = os.path.basename(d)
        info = parse_run_dir_name(name)
        if not all(k in info for k in ('h', 'dt', 'scheme', 'mass', 'p', 'mx', 'my')):
            continue
        if info.get('bc', 'homogeneous') != 'homogeneous':
            continue
        if h_filter is not None and abs(info['h'] - h_filter) > 1e-12:
            continue
        if p_filter is not None and info['p'] != p_filter:
            continue

        probe_csvs = glob.glob(os.path.join(d, 'probe-*.csv'))
        if not probe_csvs:
            continue
        probe_csv = probe_csvs[0]

        t_all, u1_all, u2_all = read_probe_csv(probe_csv)
        t, sig = choose_probe_signal(t_all, u1_all, u2_all)
        if len(t) < 16:
            continue

        meta = read_meta_csv(os.path.join(d, 'run-meta.csv'))
        h_min = meta.get('h_min', math.sqrt(2.0) * info['h'])
        cfl = meta.get('cfl', info['dt'] / h_min)

        mx = info['mx']
        my = info['my']
        omega_exact = np.pi * np.sqrt(mx * mx + my * my)
        kh = omega_exact * h_min
        if kh < min_kh:
            continue

        omega_num = estimate_omega(t, sig)
        if not np.isfinite(omega_num) or omega_exact <= 0.0:
            continue
        ratio = omega_num / omega_exact

        G = estimate_amplification_per_step(sig)

        if not (np.isfinite(ratio) and ratio > 0.0):
            continue
        # Drop obvious outliers from unstable/aliased runs in the dispersion plot.
        if ratio > 2.0:
            continue

        records.append({
            'scheme': info['scheme'],
            'mass': info['mass'],
            'h': info['h'],
            'p': info['p'],
            'mx': mx,
            'my': my,
            'cfl': cfl,
            'kh': kh,
            'ratio': ratio,
            'G': G,
        })

    return records


def make_dispersion_figure(records,
                           out_dir,
                           cfl_targets,
                           cd_cfl_targets,
                           semi_cfl,
                           cfl_tol):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: semi-discrete approximation from smallest-CFL runs.
    ax = axes[0]
    panel_curves = []
    for mass, color in [('lumped', '#2196F3'), ('consistent', '#E91E63')]:
        cand = [r for r in records if r['mass'] == mass]
        if not cand:
            continue
        # Prefer user-provided semi_cfl; fallback to minimum available CFL.
        near = [r for r in cand if abs(r['cfl'] - semi_cfl) <= cfl_tol]
        if near:
            use_cfl = semi_cfl
            chosen = near
        else:
            use_cfl = min(r['cfl'] for r in cand)
            chosen = [r for r in cand if abs(r['cfl'] - use_cfl) <= cfl_tol]

        chosen = dedup_by_kh(chosen)
        if len(chosen) < 2:
            continue
        x = [r['kh'] for r in chosen]
        y = [r['ratio'] for r in chosen]
        panel_curves.append(y)
        ax.plot(x, y, '-', color=color, linewidth=2,
                label=f'{mass}, CFL≈{use_cfl:.3g}')

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5,
               label='Exact')
    ax.set_xlabel('$kh$', fontsize=13)
    ax.set_ylabel('$\\omega_h^d/\\omega_{\\rm exact}$', fontsize=13)
    ax.set_title('Low-CFL numerical dispersion', fontsize=13)
    ax.set_xlim(left=1.0)
    ax.grid(True, alpha=0.3)
    set_dynamic_ylim(ax, panel_curves, include_values=[1.0])
    ax.legend(fontsize=10)

    # Panel 2: central difference
    ax = axes[1]
    panel_curves = []
    for i, cfl_t in enumerate(cd_cfl_targets):
        color = ['#2196F3', '#4CAF50', '#FF9800'][i % 3]
        for mass, style in [('lumped', '-'), ('consistent', '--')]:
            subset = [r for r in records
                      if r['scheme'] == 'cd'
                      and r['mass'] == mass
                      and abs(r['cfl'] - cfl_t) <= cfl_tol]
            subset = dedup_by_kh(subset)
            if len(subset) < 2:
                continue
            x = [r['kh'] for r in subset]
            y = [r['ratio'] for r in subset]
            panel_curves.append(y)
            ax.plot(x, y, linestyle=style, color=color, linewidth=1.6,
                    label=f'CFL={cfl_t}, {mass}')

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('$kh$', fontsize=13)
    ax.set_ylabel('$\\omega_h^d/\\omega_{\\rm exact}$', fontsize=13)
    ax.set_title('Central difference', fontsize=13)
    ax.set_xlim(left=1.0)
    ax.grid(True, alpha=0.3)
    set_dynamic_ylim(ax, panel_curves, include_values=[1.0])
    ax.legend(fontsize=8, ncol=2)

    # Panel 3: Newmark
    ax = axes[2]
    panel_curves = []
    for i, cfl_t in enumerate(cfl_targets):
        color = ['#2196F3', '#4CAF50', '#FF9800'][i % 3]
        for mass, style in [('lumped', '-'), ('consistent', '--')]:
            subset = [r for r in records
                      if r['scheme'] == 'newmark'
                      and r['mass'] == mass
                      and abs(r['cfl'] - cfl_t) <= cfl_tol]
            subset = dedup_by_kh(subset)
            if len(subset) < 2:
                continue
            x = [r['kh'] for r in subset]
            y = [r['ratio'] for r in subset]
            panel_curves.append(y)
            ax.plot(x, y, linestyle=style, color=color, linewidth=1.6,
                    label=f'CFL={cfl_t}, {mass}')

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('$kh$', fontsize=13)
    ax.set_ylabel('$\\omega_h^d/\\omega_{\\rm exact}$', fontsize=13)
    ax.set_title('Newmark ($\\beta=1/4, \\gamma=1/2$)', fontsize=13)
    ax.set_xlim(left=1.0)
    ax.grid(True, alpha=0.3)
    set_dynamic_ylim(ax, panel_curves, include_values=[1.0])
    ax.legend(fontsize=8, ncol=2)

    fig.suptitle('Dispersion relation from simulation data', fontsize=15, y=0.98)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

    out = os.path.join(out_dir, 'dispersion_relation.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')


def make_amplification_figure(records,
                              out_dir,
                              scheme,
                              cfl_targets,
                              cfl_tol,
                              mass_types):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5,
               label='Exact (no dissipation)')

    plotted = 0
    for i, cfl_t in enumerate(cfl_targets):
        color = ['#2196F3', '#4CAF50', '#FF9800'][i % 3]
        for mass, style in [('lumped', '-'), ('consistent', '--')]:
            if mass not in mass_types:
                continue
            subset = [r for r in records
                      if r['scheme'] == scheme
                      and r['mass'] == mass
                      and abs(r['cfl'] - cfl_t) <= cfl_tol
                      and np.isfinite(r['G'])]
            subset = dedup_by_kh(subset)
            if len(subset) < 2:
                continue
            x = [r['kh'] for r in subset]
            y = [r['G'] for r in subset]
            ax.plot(x, y, linestyle=style, color=color, linewidth=1.5,
                    label=f'CFL={cfl_t}, {mass}')
            plotted += 1

    ax.set_xlabel('$kh$', fontsize=13)
    ax.set_ylabel('Amplification factor $|G|$', fontsize=13)
    ax.set_xlim(left=1.0)
    ax.set_ylim(0.8, 1.2)
    if scheme == 'cd':
        ax.set_title('Central difference\namplification per step', fontsize=13)
        out_name = 'dissipation_amplification_central.png'
    else:
        ax.set_title('Newmark ($\\beta=1/4, \\gamma=1/2$)\namplification per step', fontsize=13)
        out_name = 'dissipation_amplification_newmark.png'

    ax.grid(True, alpha=0.3)
    # Force plain absolute values to avoid confusing offset text such as
    # "1e-6 + 9.9999e-1" in the report figure.
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.6f'))
    ax.legend(fontsize=8)

    out = os.path.join(out_dir, out_name)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    if plotted == 0:
        print(f'[info] No valid amplification curves for scheme={scheme}.')


def main():
    raw = [a for a in sys.argv[1:] if a != '--show']
    show_plot = '--show' in sys.argv
    if not raw:
        print('Usage: python3 plot_dispersion_from_results.py <result_dir> '
              '[--h 0.05] [--p 1] [--cfl-values 0.5,0.8,1.0] '
              '[--cd-cfl-values 0.1] [--amp-cfl-values 0.1] '
              '[--amp-mass-types lumped,consistent] [--semi-cfl 0.1] '
              '[--min-kh 1.0]')
        sys.exit(1)

    result_dir = raw[0]
    h_filter = None
    p_filter = 1
    cfl_targets = [0.5, 0.8, 1.0]
    cd_cfl_targets = None
    amp_cfl_targets = None
    amp_mass_types = ['lumped', 'consistent']
    semi_cfl = 0.1
    cfl_tol = 0.03
    min_kh = 1.0

    i = 1
    while i < len(raw):
        a = raw[i]
        if a == '--h' and i + 1 < len(raw):
            h_filter = float(raw[i + 1])
            i += 2
            continue
        if a == '--p' and i + 1 < len(raw):
            p_filter = int(raw[i + 1])
            i += 2
            continue
        if a == '--cfl-values' and i + 1 < len(raw):
            cfl_targets = [float(x) for x in raw[i + 1].split(',') if x.strip()]
            i += 2
            continue
        if a == '--cd-cfl-values' and i + 1 < len(raw):
            cd_cfl_targets = [float(x) for x in raw[i + 1].split(',') if x.strip()]
            i += 2
            continue
        if a == '--amp-cfl-values' and i + 1 < len(raw):
            amp_cfl_targets = [float(x) for x in raw[i + 1].split(',') if x.strip()]
            i += 2
            continue
        if a == '--amp-mass-types' and i + 1 < len(raw):
            amp_mass_types = [x.strip() for x in raw[i + 1].split(',') if x.strip()]
            i += 2
            continue
        if a == '--semi-cfl' and i + 1 < len(raw):
            semi_cfl = float(raw[i + 1])
            i += 2
            continue
        if a == '--cfl-tol' and i + 1 < len(raw):
            cfl_tol = float(raw[i + 1])
            i += 2
            continue
        if a == '--min-kh' and i + 1 < len(raw):
            min_kh = float(raw[i + 1])
            i += 2
            continue
        i += 1

    records = collect_records(result_dir, h_filter, p_filter, min_kh)
    if len(records) < 10:
        print('[info] Not enough probe-based runs found for data-driven '
              'dispersion plots.')
        print('       Run scripts/run_dispersion_experiments.sh first.')
        return

    if cd_cfl_targets is None:
        cd_cfl_targets = cfl_targets
    if amp_cfl_targets is None:
        amp_cfl_targets = cfl_targets

    make_dispersion_figure(records,
                           result_dir,
                           cfl_targets,
                           cd_cfl_targets,
                           semi_cfl,
                           cfl_tol)
    make_amplification_figure(records,
                              result_dir,
                              'cd',
                              amp_cfl_targets,
                              cfl_tol,
                              amp_mass_types)
    make_amplification_figure(records,
                              result_dir,
                              'newmark',
                              amp_cfl_targets,
                              cfl_tol,
                              amp_mass_types)

    if show_plot:
        plt.show()


if __name__ == '__main__':
    main()

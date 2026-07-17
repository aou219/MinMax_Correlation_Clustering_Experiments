#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, os, shutil, sys, tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
for p in (REPO_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.edge_deletion import delete_edges
from src.experiment_helpers import compute_min_max_cc_data
from src.facebook_sampling import (
    build_complete_signed_matrix_from_facebook_sample,
    load_facebook_circles,
    load_facebook_ego_edges,
)

DEFAULT_GRID = REPO_ROOT / "results/processed/research_tables/minmax_facebook_grid_runs_flat.csv"


def parse_int_spec(raw: str) -> list[int]:
    out = []
    for part in raw.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = map(int, part.split('-', 1))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_float_list(raw: str) -> list[float]:
    return sorted(set(float(x.strip()) for x in raw.split(',') if x.strip()))


def locate(ego: str, ext: str) -> Path:
    for p in (
        REPO_ROOT / f"data/facebook/{ego}.{ext}",
        REPO_ROOT / f"data/facebook/facebook_3/{ego}.{ext}",
    ):
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing Facebook {ext} file for ego {ego}")


def reconstruct(ego: str) -> np.ndarray:
    edge_nodes, edges = load_facebook_ego_edges(str(locate(ego, 'edges')))
    circles = load_facebook_circles(str(locate(ego, 'circles')))
    circle_nodes = {v for c in circles for v in c['nodes']}
    nodes = sorted(edge_nodes | circle_nodes)
    matrix, _, _, _ = build_complete_signed_matrix_from_facebook_sample(nodes, edges)
    return matrix


def f8(x: Any) -> str:
    return f"{float(x):.8f}"


def key(ego: str, p: Any, seed: Any, d: Any, lam: Any):
    return (str(ego), f8(p), str(int(float(seed))), str(int(float(d))), str(int(float(lam))))


def atomic_write(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent, text=True)
    try:
        with os.fdopen(fd, 'w', newline='', encoding='utf-8') as h:
            w = csv.DictWriter(h, fieldnames=fields, extrasaction='ignore')
            w.writeheader(); w.writerows(rows)
        os.replace(tmp, path)
    except Exception:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise


def put_cc(row: dict[str, str], prefix: str, result: dict[str, Any], d: int, lam: int) -> None:
    row[f"{prefix}_min_max_cc_computed"] = str(result['max_disagreement'] is not None)
    row[f"{prefix}_min_max_cc_cluster_count"] = str(result['cluster_count'])
    row[f"{prefix}_min_max_cc_max_disagreement"] = str(result['max_disagreement'])
    row[f"{prefix}_min_max_cc_d_hat"] = str(d)
    row[f"{prefix}_min_max_cc_lambda"] = str(lam)
    row[f"{prefix}_min_max_cc_runtime_seconds"] = str(result['runtime_seconds'])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', type=Path, default=DEFAULT_GRID)
    ap.add_argument('--lambda-values', default='5-20')
    ap.add_argument('--seeds', default='1-30')
    ap.add_argument('--ego-order', default='3980,698,414,686')
    ap.add_argument('--p-delete-values', default='0.05,0.15,0.25,0.4')
    ap.add_argument('--no-backup', action='store_true')
    args = ap.parse_args()

    grid = args.grid if args.grid.is_absolute() else REPO_ROOT / args.grid
    lambdas = parse_int_spec(args.lambda_values)
    if any(l <= 4 for l in lambdas):
        raise ValueError('For q=0, lambda must be > 4.')
    seeds = parse_int_spec(args.seeds)
    egos = [x.strip() for x in args.ego_order.split(',') if x.strip()]
    p_values = parse_float_list(args.p_delete_values)

    with grid.open(newline='', encoding='utf-8-sig') as h:
        r = csv.DictReader(h)
        fields = list(r.fieldnames or [])
        rows = [dict(x) for x in r]

    cm = {x.get('complete_min_max_lp_method','').strip() for x in rows if x.get('complete_min_max_lp_method','').strip()}
    em = {x.get('edge_min_max_lp_method','').strip() for x in rows if x.get('edge_min_max_lp_method','').strip()}
    if cm != {'2'} or em != {'2'}:
        raise ValueError(f'Expected only method 2. complete={cm}, edge={em}')

    reps = {}
    d_by_ego = defaultdict(set)
    done = set()
    for row in rows:
        ego = row['ego_id'].strip(); p = row['p_delete'].strip(); seed = row.get('seed','1') or '1'
        reps.setdefault((ego, f8(p), str(int(float(seed)))), row)
        d = row.get('edge_min_max_cc_d_hat','').strip(); lam = row.get('edge_min_max_cc_lambda','').strip()
        if d: d_by_ego[ego].add(int(float(d)))
        if d and lam: done.add(key(ego, p, seed, d, lam))

    if not args.no_backup:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = grid.with_name(f"{grid.stem}_before_extra_lambdas_{stamp}{grid.suffix}")
        shutil.copy2(grid, backup)
        print('Backup:', backup)

    matrix_cache = {}
    complete_cache = {}
    added = 0

    for ego in egos:
        matrix_cache[ego] = reconstruct(ego)
        complete = matrix_cache[ego]
        d_values = sorted(d_by_ego[ego])
        for p in p_values:
            for seed in seeds:
                missing = [(d, lam) for d in d_values for lam in lambdas if key(ego,p,seed,d,lam) not in done]
                if not missing:
                    continue
                print(f"\nego={ego} p_delete={p} seed={seed} missing={len(missing)}")
                edge, deleted = delete_edges(complete, p, seed)
                print('Deleted edges:', deleted)
                rep = reps[(ego, f8(p), str(seed))]
                for d, lam in missing:
                    ck = (ego, d, lam)
                    if ck not in complete_cache:
                        print(f"complete CC d_hat={d} lambda={lam}")
                        complete_cache[ck] = compute_min_max_cc_data(complete, True, d, lam)
                    print(f"edge CC d_hat={d} lambda={lam}")
                    edge_cc = compute_min_max_cc_data(edge, True, d, lam)
                    new = {c: rep.get(c,'') for c in fields}
                    put_cc(new, 'complete', complete_cache[ck], d, lam)
                    put_cc(new, 'edge', edge_cc, d, lam)
                    rows.append(new); done.add(key(ego,p,seed,d,lam)); added += 1
                    rows.sort(
                        key=lambda row: (
                            int(float(row.get("ego_id", 0) or 0)),
                            float(row.get("p_delete", 0) or 0),
                            int(float(row.get("seed", 1) or 1)),
                            int(
                                float(
                                    row.get(
                                        "edge_min_max_cc_d_hat",
                                        0,
                                    )
                                    or 0
                                )
                            ),
                            int(
                                float(
                                    row.get(
                                        "edge_min_max_cc_lambda",
                                        0,
                                    )
                                    or 0
                                )
                            ),
                        )
                    )
                    atomic_write(grid, fields, rows)
                    print('Saved. New rows:', added)

    print('\nFinished. Added rows:', added)
    print('Output:', grid)

if __name__ == '__main__':
    main()

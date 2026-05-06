"""Build a UUID -> benchmark_name map from molmo_spaces eval output.

Walks the molmo_spaces benchmark output tree and reads each trajectory's
`obs_scene` (a JSON blob in the .h5 file) to extract the tiptop `run_id`
that was sent on the websocket request. The leaf directory name (e.g.
``pnp_next_to-v2_20260427_162204``) is used as the benchmark label.

The result is a flat JSON map ``{run_id: benchmark_name}`` that can be
passed to ``analyze_results.py --benchmark-map``.
"""

import argparse
import json
from pathlib import Path

import h5py

DEFAULT_GOOD_EVALS = Path(
    "/home/willshen/projects/ml/robotics/molmospaces/eval_output/TiptopPolicyEvalConfig/good_evals"
)


def extract_run_ids_from_h5(h5_path: Path) -> list[str]:
    """Return all tiptop run_ids found across traj_* groups in *h5_path*."""
    run_ids: list[str] = []
    with h5py.File(h5_path, "r") as f:
        for traj_name in f.keys():
            if not traj_name.startswith("traj_"):
                continue
            traj = f[traj_name]
            if "obs_scene" not in traj:
                continue
            scene = traj["obs_scene"]
            val = scene[()] if scene.shape == () else scene[0]
            if isinstance(val, bytes):
                val = val.decode()
            try:
                meta = json.loads(val)
            except json.JSONDecodeError:
                continue
            run_id = meta.get("run_id")
            if run_id:
                run_ids.append(run_id)
    return run_ids


def build_map(root: Path) -> dict[str, str]:
    """Build run_id -> benchmark_name map by walking *root*.

    A "benchmark" is the immediate child directory of *root* (e.g.
    ``pnp_next_to-v2_20260427_162204``). Any .h5 file beneath it is
    attributed to that benchmark.
    """
    mapping: dict[str, str] = {}
    collisions: list[tuple[str, str, str]] = []  # (run_id, existing, new)

    for benchmark_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        benchmark_name = benchmark_dir.name
        h5_files = list(benchmark_dir.rglob("*.h5"))
        if not h5_files:
            print(f"[skip] {benchmark_name}: no .h5 files")
            continue

        run_ids_in_bench = 0
        for h5_path in h5_files:
            try:
                run_ids = extract_run_ids_from_h5(h5_path)
            except (OSError, KeyError) as e:
                print(f"[warn] failed to read {h5_path}: {e}")
                continue
            for run_id in run_ids:
                if run_id in mapping and mapping[run_id] != benchmark_name:
                    collisions.append((run_id, mapping[run_id], benchmark_name))
                mapping[run_id] = benchmark_name
                run_ids_in_bench += 1

        print(f"[ok]   {benchmark_name}: {len(h5_files)} h5 files, {run_ids_in_bench} run_ids")

    if collisions:
        print(f"\n[warn] {len(collisions)} run_ids appeared under multiple benchmarks (kept last seen):")
        for run_id, existing, new in collisions[:10]:
            print(f"  {run_id}: {existing} -> {new}")
        if len(collisions) > 10:
            print(f"  ... and {len(collisions) - 10} more")

    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_GOOD_EVALS,
        help=f"molmo_spaces good_evals directory (default: {DEFAULT_GOOD_EVALS})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_map.json"),
        help="Output JSON path (default: benchmark_map.json)",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"Root directory not found: {args.root}")

    mapping = build_map(args.root)

    with open(args.output, "w") as f:
        json.dump(mapping, f, indent=2, sort_keys=True)

    print(f"\nWrote {len(mapping)} run_id -> benchmark mappings to {args.output}")


if __name__ == "__main__":
    main()

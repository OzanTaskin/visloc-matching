"""Run the benchmark: build maps, localize queries under each condition, report.

Examples:
    python run.py                                  # full benchmark on ShopFacade
    python run.py --max-queries 10 --methods superpoint-lightglue --conditions clean
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import visloc  # noqa: E402,F401  (sets the OpenMP flag before torch/pycolmap load)
import pandas as pd  # noqa: E402

from visloc import pipeline, report, visualize  # noqa: E402
from visloc.config import (CONDITIONS, DEFAULT_SCENE, FIGURES_DIR, METHODS,  # noqa: E402
                           RESULTS_DIR, scene_paths)
from visloc.evaluate import evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                        choices=list(CONDITIONS))
    parser.add_argument("--max-queries", type=int, default=0,
                        help="Use only the first N queries (quick smoke test).")
    args = parser.parse_args()

    smoke = args.max_queries > 0
    p = scene_paths(args.scene, tag=f"-q{args.max_queries}" if smoke else "")
    results_dir = p["out"] / "results" if smoke else RESULTS_DIR
    queries = pipeline.prepare_scene(p, args.max_queries)
    print(f"{args.scene}: {len(queries)} queries, methods={args.methods}, "
          f"conditions={args.conditions}")

    summaries, per_query, runs = [], [], {}
    for name in args.methods:
        method = METHODS[name]
        map_paths = pipeline.build_map(p, method)
        for cond in args.conditions:
            run = pipeline.localize(p, method, map_paths, cond, queries)
            rows, summary = evaluate(run["poses"], p["gt"] / "images.txt", queries,
                                     failed=run["failed"])
            summary |= {"scene": args.scene, "method": name, "condition": cond,
                        "seconds_per_query": run["seconds_per_query"]}
            (run["poses"].parent / "metrics.json").write_text(json.dumps(summary, indent=2))
            print(f"  {method.label:<24} {cond:<7} "
                  f"median {summary['median_t_cm']:.1f} cm / {summary['median_R_deg']:.2f} deg, "
                  f"25cm-2deg {summary['recall_25cm_2deg']:.1f}%")
            summaries.append(summary)
            per_query += [r | {"method": name, "condition": cond} for r in rows]
            runs[(name, cond)] = run

    results_dir.mkdir(parents=True, exist_ok=True)
    report.make_all(pd.DataFrame(summaries), pd.DataFrame(per_query), results_dir)

    figures = results_dir / "figures"
    visualize.plot_conditions(p["images"] / queries[0], queries[0], figures / "conditions.png")
    hardest = "dark" if "dark" in args.conditions else args.conditions[0]
    shown = [m for m in ("sift-nn", "superpoint-lightglue") if m in args.methods]
    if shown:
        visualize.plot_topdown(
            p["out"] / shown[-1] / "sfm", p["gt"] / "images.txt",
            {METHODS[m].label: runs[(m, hardest)]["poses"] for m in shown}, queries,
            figures / f"topdown_{hardest}.png",
            failed={METHODS[m].label: runs[(m, hardest)]["failed"] for m in shown},
            title=f"Ground truth (green) to estimate (red), condition: {hardest}")
        visualize.export_3d(p["out"] / shown[-1] / "sfm", p["out"] / "map_3d.html")
    print(f"Results written to {results_dir}")


if __name__ == "__main__":
    main()

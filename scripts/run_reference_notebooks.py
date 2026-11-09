"""Execute the curated reference notebooks and collect their check results.

Runs inside the optimuslib/optimus Docker image (Python 3.5), so no f-strings.

Usage (from the repository root, inside the container):

    python3 scripts/run_reference_notebooks.py --ci                 # all notebooks, CI mode
    python3 scripts/run_reference_notebooks.py 01_basic_model       # one notebook, full mode

Each notebook runs with the working directory notebooks/reference and with the
repository root first on PYTHONPATH, so the repository's own optimus package is
tested rather than the copy installed in the image. Results go to --output-dir
(default: reference-run): executed notebooks, figures, one JSON file per
notebook, and a summary (validated_notebooks.json, summary.md).
"""

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
import time

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTEBOOK_DIR = os.path.join(ROOT, "notebooks", "reference")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")


def run_one(path, output_dir, ci_mode, timeout):
    name = os.path.splitext(os.path.basename(path))[0]
    nb_output_dir = os.path.join(output_dir, name)
    if not os.path.isdir(nb_output_dir):
        os.makedirs(nb_output_dir)

    os.environ["OPTIMUS_CI_MODE"] = "1" if ci_mode else "0"
    os.environ["OPTIMUS_OUTPUT_DIR"] = nb_output_dir
    os.environ["PYTHONPATH"] = ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")

    nb = nbformat.read(path, as_version=4)
    executor = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    status, error = "passed", None
    start = time.time()
    try:
        executor.preprocess(nb, {"metadata": {"path": NOTEBOOK_DIR}})
    except CellExecutionError as exc:
        status = "failed"
        error = str(exc).strip().splitlines()[-1][:500]
    except Exception as exc:  # timeouts, dead kernels
        status = "error"
        error = repr(exc)[:500]
    runtime = time.time() - start
    nbformat.write(nb, os.path.join(output_dir, name + ".executed.ipynb"))

    result = {"notebook": name, "status": status, "runtime_s": round(runtime, 1),
              "ci_mode": ci_mode, "error": error, "checks": {}}
    results_file = os.path.join(nb_output_dir, name + "_results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            details = json.load(f)
        result["checks"] = details.get("checks", {})
        result["details"] = details
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebooks", nargs="*", help="names (without .ipynb); default: all")
    parser.add_argument("--ci", action="store_true", help="run with OPTIMUS_CI_MODE=1")
    parser.add_argument("--output-dir", default=os.path.join(ROOT, "reference-run"))
    parser.add_argument("--timeout", type=int, default=900, help="per-cell timeout in seconds")
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(NOTEBOOK_DIR, "[0-9][0-9]_*.ipynb")))
    if args.notebooks:
        wanted = set(n.replace(".ipynb", "") for n in args.notebooks)
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in wanted]
        if not paths:
            sys.exit("No matching notebooks for: " + ", ".join(sorted(wanted)))

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    results = []
    for path in paths:
        print("Running " + os.path.basename(path) + (" (CI mode)" if args.ci else "") + " ...")
        sys.stdout.flush()
        result = run_one(path, output_dir, args.ci, args.timeout)
        failed_checks = [k for k, v in result["checks"].items() if not v]
        print("  %s in %.1f s, %d checks passed%s" % (
            result["status"].upper(), result["runtime_s"],
            sum(1 for v in result["checks"].values() if v),
            (", failed: " + ", ".join(failed_checks)) if failed_checks else ""))
        if result["error"]:
            print("  " + result["error"])
        sys.stdout.flush()
        results.append(result)

    summary = {
        "commit": git_commit(),
        "date": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ci_mode": args.ci,
        "notebooks": results,
    }
    with open(os.path.join(output_dir, "validated_notebooks.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    lines = ["| Notebook | Status | Runtime (s) | Checks passed |", "|---|---|---|---|"]
    for r in results:
        lines.append("| %s | %s | %.1f | %d / %d |" % (
            r["notebook"], r["status"], r["runtime_s"],
            sum(1 for v in r["checks"].values() if v), len(r["checks"])))
    table = "\n".join(lines) + "\n"
    with open(os.path.join(output_dir, "summary.md"), "w") as f:
        f.write(table)
    print("\n" + table)

    sys.exit(0 if all(r["status"] == "passed" for r in results) else 1)


if __name__ == "__main__":
    main()

# Curated reference notebooks

Eight notebooks that cover the common OptimUS workflows. Each one is adapted from an upstream
notebook in `notebooks/`, documents its purpose, inputs and outputs in a header table, and ends with
**checks**: assertions on physics or numerics that make the notebook fail if the answer is wrong.
They are run automatically by `.github/workflows/reference-notebooks.yml` and are the trusted
source set for the project's documentation-grounded assistant.

| # | Notebook | Why it is in the set | What the checks test |
|---|---|---|---|
| 01 | [`01_basic_model.ipynb`](01_basic_model.ipynb) | The five-step workflow every simulation follows; the first thing a new user needs. | GMRES converged; unit incident amplitude; total field within 5% of the analytical sphere solution. |
| 02 | [`02_piston_source.ipynb`](02_piston_source.ipynb) | Transducer sources, placement and aiming; how sources are sampled with point sources. | On-axis pressure vs the analytical piston formula (Kinsler et al. 2000), and convergence as the source is refined. |
| 03 | [`03_bowl_source.ipynb`](03_bowl_source.ipynb) | The standard focused ultrasound transducer. | On-axis pressure converges to O'Neil (1949); pre-focal shift of the peak; documents that `location` is the bowl apex. |
| 04 | [`04_array_source.ipynb`](04_array_source.ipynb) | Multi-element therapeutic arrays at realistic size and frequency (500 kHz, 256 elements). | Focus within one wavelength of the geometric focus; -6 dB width within 20% of 1.41 lambda F#. |
| 05 | [`05_material_properties.ipynb`](05_material_properties.ipynb) | Every simulation needs materials; covers the database, new materials and saving them. | Database values, wavenumber and attenuation definitions, energy conservation at an interface, save and reload, duplicate names rejected, database restored. |
| 06 | [`06_nested_domains.ipynb`](06_nested_domains.ipynb) | Domains inside domains, the pattern for tissue models; wavelength-based mesh sizing. | Graph topology; GMRES converged; pressure inside an air pocket under 1% of the incident amplitude. |
| 07 | [`07_validation.ipynb`](07_validation.ipynb) | The evidence that OptimUS converges to the exact answer. | Error decreases with every refinement; finest mesh under 1% (full) or 2% (CI); convergence order at least 1. |
| 08 | [`08_visualisation_and_postprocess.ipynb`](08_visualisation_and_postprocess.ipynb) | Everything after the solve: planes, zooms, point clouds, custom plots, export and import. | Point-cloud field equals plane field; total = incident + scattered; exported solution reloads unchanged. |

Upstream notebooks left out of the core set (still in `notebooks/`): Acoustic intensity and power,
Array source configuration, BioHeat, Bowl source aberrated by a skull slab, Focusing with phase
conjugation, Global parameters, In utero sound transmission, Incident field parallelisation, the two
Line array notebooks, Multiple domains, Object shapes, Preconditioners, Time domain simulations,
Validation with two concentric spheres, Visualisation in 3D. Most extend one of the eight above;
some (skull slab, in utero) use large meshes that are too slow for CI.

## Measured runtimes

Measured on 9 Nov 2026 in the `optimuslib/optimus:latest` Docker image (Python 3.5.2, 4 CPUs,
15 GB RAM), whole notebook including plots and checks.

| Notebook | Full mode (s) | CI mode (s) | Checks | Key measured numbers (full mode) |
|---|---|---|---|---|
| 01_basic_model | 32 | 15 | 4 | 118 vertices, 25 GMRES iterations, 2.7% error vs analytical |
| 02_piston_source | 26 | 17 | 4 | on-axis error 6.6% (6 points per wavelength), 1.2% (24) |
| 03_bowl_source | 27 | 14 | 5 | on-axis error 8.3% (6), 0.5% (24); peak 0.32 m from apex, focus at 1 m |
| 04_array_source | 177 | 69 | 4 | 1607 vertices, 64 iterations, 77 s solve; focus 1.7 mm off, -6 dB width 4.5 mm (estimate 4.8) |
| 05_material_properties | 2 | 3 | 7 | R water to bone 0.64, water to fat -0.06 |
| 06_nested_domains | 40 | 28 | 4 | 615 vertices, 199 iterations; max \|p\| in air 0.0025 |
| 07_validation | 257 | 30 | 4 | error 10.6% (h = 1.0) down to 0.81% (h = 0.1), order 1.04 |
| 08_visualisation_and_postprocess | 25 | 17 | 4 | 24 iterations; cloud and plane fields identical |
| **Total** | **586 (about 10 min)** | **194 (about 3 min)** | **36** | |

## Do the checks catch wrong answers?

`scripts/check_planted_faults.py` plants six known faults in a copy of the repository and runs the
affected notebook in CI mode. On 9 Nov 2026 all six were caught:

| Planted fault | Notebook | Check that failed |
|---|---|---|
| Plane wave normal derivative loses its factor i (library) | 01 | `matches_analytical` (error 48%) |
| Mesh far too coarse, element size 1.5 m (notebook) | 01 | `matches_analytical` (error 10%) |
| Rayleigh integral prefactor halved (library) | 02 | `source_default_accuracy` (error 49%) |
| Bowl radius of curvature stored 20% too large (library) | 03 | `source_default_accuracy` (error 26%) |
| Wavelength computed as f / c (library) | 05 | `wavenumber_definition` |
| GMRES tolerance loosened from 1e-5 to 1e-1 (library) | 07 | `error_decreases` (errors 0.101, 0.072, 0.040, 0.051) |

The last one is a useful lesson: GMRES still reports convergence with a loose tolerance, so the
`gmres_converged` check alone would not catch it; the comparison with the analytical solution does.

```powershell
docker run --rm -v "${PWD}:/work" -w /work optimuslib/optimus:latest python3 scripts/check_planted_faults.py
```

## Conventions every reference notebook follows

* **Header table** with Purpose, Inputs, Outputs, Checks, Runtime and Adapted from.
* **Physics notes**, then a *To expand* box: the engineering derivations to be written up by Yash.
* **`CI_MODE`**: read from the environment variable `OPTIMUS_CI_MODE` (`1` = on). CI mode shrinks
  plotting grids and, where needed, meshes, so the notebook runs in about a minute. Physics settings
  that the checks depend on stay the same.
* **`OUTPUT_DIR`**: from `OPTIMUS_OUTPUT_DIR`, default `outputs/`. Figures are saved there, plus
  `<notebook>_results.json` with the key numbers and the result of every check.
* **`check(name, condition, detail)`**: prints PASS or FAIL and raises `AssertionError` on failure.
  Every threshold states where it comes from: a cited formula, a project success criterion, or a
  measured value plus a margin (with the date it was measured).
* **Python 3.5 compatible** (the Docker image): no f-strings.
* **No stored outputs** in the committed notebook; no `sys.path` hacks (the runner sets `PYTHONPATH`).

`tests/test_reference_notebooks.py` checks these conventions without needing OptimUS.

## Running the notebooks

### In Jupyter

Open them from `notebooks/reference` in the Docker image's Jupyter as usual. They run in full mode
unless `OPTIMUS_CI_MODE=1` is set. Figures appear inline and are also saved to `outputs/`.

### All of them, from PowerShell (Windows, Docker Desktop)

From the repository folder (for example `C:\Users\reach\OptimUS\optimus`):

```powershell
docker run --rm -v "${PWD}:/work" -w /work optimuslib/optimus:latest python3 scripts/run_reference_notebooks.py --ci --output-dir /work/reference-run
```

Leave out `--ci` for full mode, or name notebooks to run only those:

```powershell
docker run --rm -v "${PWD}:/work" -w /work optimuslib/optimus:latest python3 scripts/run_reference_notebooks.py 07_validation --output-dir /work/reference-run
```

Results land in `reference-run\`: executed notebooks, figures, `summary.md` and
`validated_notebooks.json` (notebook, commit, date, status, every check). The harmless
"Unexpected end of /proc/mounts line" warning can be ignored.

### Static checks only (any Python 3)

```powershell
python -m pip install pytest
python -m pytest tests/test_reference_notebooks.py -v
```

## Adding or changing a notebook

1. Copy the closest existing reference notebook and keep its structure.
2. Set `NOTEBOOK_NAME` to the file name without `.ipynb`.
3. Add at least three `check(...)` calls, and write in the Checks markdown where each threshold comes from.
4. Clear all outputs before committing.
5. Add the notebook to the table above and run the static checks and the runner.

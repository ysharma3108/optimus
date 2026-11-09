"""Plant known faults and confirm the reference notebooks catch every one.

Each fault is a one-line change to a copy of the repository (the library or a
notebook). The script runs the affected notebooks in CI mode on that copy and
passes only if each fault makes the expected check fail. This is the evidence
that a wrong result turns CI red (project success criterion N3).

Runs inside the optimuslib/optimus Docker image (Python 3.5), from the
repository root:

    python3 scripts/check_planted_faults.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (description, file, original text, faulty text, notebook, check expected to fail)
FAULTS = [
    ("Incident plane wave: normal derivative loses its factor i",
     "optimus/source/planewave.py",
     "(self.amplitude * 1j * wavenumber)",
     "(self.amplitude * wavenumber)",
     "01_basic_model", "matches_analytical"),
    ("Basic model: mesh far too coarse (element size 1.5 m, about 1 per wavelength)",
     "notebooks/reference/01_basic_model.ipynb",
     "element_size = 0.4 ",
     "element_size = 1.5 ",
     "01_basic_model", "matches_analytical"),
    ("Transducer sources: Rayleigh integral prefactor halved",
     "optimus/source/transducers.py",
     "(2j * _np.pi * frequency * density) * values",
     "(1j * _np.pi * frequency * density) * values",
     "02_piston_source", "source_default_accuracy"),
    ("Bowl source: radius of curvature stored 20% too large",
     "optimus/source/bowl.py",
     "self.radius_of_curvature = _convert_to_positive_float(radius_of_curvature)",
     "self.radius_of_curvature = 1.2 * _convert_to_positive_float(radius_of_curvature)",
     "03_bowl_source", "source_default_accuracy"),
    ("Materials: wavelength computed as f / c instead of c / f",
     "optimus/material/common.py",
     "return self.speed_of_sound / frequency",
     "return frequency / self.speed_of_sound",
     "05_material_properties", "wavenumber_definition"),
    ("Solver: GMRES tolerance loosened from 1e-5 to 1e-1",
     "optimus/utils/parameters.py",
     "self.tol = 1e-5",
     "self.tol = 1e-1",
     "07_validation", "error_decreases"),
]


def run_fault(fault):
    description, rel_path, original, faulty, notebook, expected = fault
    workdir = tempfile.mkdtemp(prefix="planted_fault_")
    try:
        for item in ("optimus", "notebooks", "scripts"):
            shutil.copytree(os.path.join(ROOT, item), os.path.join(workdir, item))
        path = os.path.join(workdir, rel_path)
        with open(path) as f:
            text = f.read()
        if original not in text:
            return False, "original text not found in " + rel_path
        with open(path, "w") as f:
            f.write(text.replace(original, faulty))

        output_dir = os.path.join(workdir, "run")
        subprocess.call([sys.executable, os.path.join(workdir, "scripts", "run_reference_notebooks.py"),
                         "--ci", notebook, "--output-dir", output_dir],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(os.path.join(output_dir, "validated_notebooks.json")) as f:
            result = json.load(f)["notebooks"][0]
        if result["status"] == "passed":
            return False, "notebook still passed"
        error = result["error"] or ""
        if expected is None:
            return True, "failed: " + error
        if (expected + " failed") in error:
            return True, "failed as expected: " + error
        return False, "failed, but not on " + expected + ": " + error
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    caught = 0
    for fault in FAULTS:
        print("Fault: " + fault[0] + " (" + fault[4] + ")")
        sys.stdout.flush()
        ok, message = run_fault(fault)
        caught += ok
        print("  " + ("CAUGHT  " if ok else "MISSED  ") + message[:300])
        sys.stdout.flush()
    print("\n%d of %d planted faults caught" % (caught, len(FAULTS)))
    sys.exit(0 if caught == len(FAULTS) else 1)


if __name__ == "__main__":
    main()

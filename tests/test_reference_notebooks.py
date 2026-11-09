"""Static checks for the curated reference notebooks.

These run in seconds on any Python 3 with pytest and need neither OptimUS nor
bempp, so they can run on every push. Executing the notebooks themselves needs
the optimuslib/optimus Docker image: see scripts/run_reference_notebooks.py.

    python -m pytest tests/test_reference_notebooks.py -v
"""

import ast
import glob
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(ROOT, "notebooks", "reference")
NOTEBOOKS = sorted(glob.glob(os.path.join(REFERENCE_DIR, "[0-9][0-9]_*.ipynb")))
HEADER_FIELDS = ["Purpose", "Inputs", "Outputs", "Checks", "Runtime", "Adapted from"]


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def source(cell):
    return "".join(cell["source"])


def code_without_magics(cell):
    """Replace IPython magics and shell escapes so the cell parses as Python.

    Only unindented lines count: an indented line starting with % is the
    continuation of a string formatting expression.
    """
    lines = []
    for line in source(cell).splitlines():
        lines.append("pass" if line.startswith(("%", "!")) else line)
    return "\n".join(lines)


def ids(paths):
    return [os.path.basename(p) for p in paths]


def test_curated_set_exists():
    assert 6 <= len(NOTEBOOKS) <= 8, "the plan is 6 to 8 curated notebooks"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_format_and_kernel(path):
    nb = load(path)
    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["name"] == "python3"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_no_stored_outputs(path):
    """Outputs are produced by CI; committed outputs make diffs unreadable."""
    for cell in load(path)["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [], "clear outputs before committing"
            assert cell["execution_count"] is None


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_header_documents_purpose_inputs_outputs(path):
    first = load(path)["cells"][0]
    assert first["cell_type"] == "markdown"
    text = source(first)
    assert text.startswith("# "), "first cell starts with the notebook title"
    for field in HEADER_FIELDS:
        assert "**" + field + "**" in text, "header table is missing " + field


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_ci_mode_and_checks_present(path):
    code = "\n".join(source(c) for c in load(path)["cells"] if c["cell_type"] == "code")
    name = os.path.splitext(os.path.basename(path))[0]
    assert 'NOTEBOOK_NAME = "' + name + '"' in code, "NOTEBOOK_NAME must match the file name"
    assert 'os.environ.get("OPTIMUS_CI_MODE"' in code
    assert "OPTIMUS_OUTPUT_DIR" in code
    assert len(re.findall(r"^\s*check\(", code, flags=re.M)) >= 3, "at least three named checks"
    assert "_results.json" in code, "the summary cell writes the results file"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_code_is_python35_compatible(path):
    """The Docker image runs Python 3.5: no f-strings, and the code must parse."""
    for i, cell in enumerate(load(path)["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse(code_without_magics(cell), filename="%s cell %d" % (path, i))
        for node in ast.walk(tree):
            assert not isinstance(node, ast.JoinedStr), "f-string in cell %d (Python 3.5 has none)" % i
            assert not isinstance(node, ast.NamedExpr), "walrus operator in cell %d" % i
            assert not (isinstance(node, ast.AnnAssign)), "variable annotation in cell %d" % i


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_no_interactive_backends_or_path_hacks(path):
    code = "\n".join(source(c) for c in load(path)["cells"] if c["cell_type"] == "code")
    assert "%matplotlib notebook" not in code, "interactive backend breaks headless runs"
    assert "sys.path.append" not in code, "the runner sets PYTHONPATH instead"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=ids(NOTEBOOKS))
def test_data_files_exist(path):
    code = "\n".join(source(c) for c in load(path)["cells"] if c["cell_type"] == "code")
    for match in re.findall(r'os\.path\.join\("\.\.", "Data", "([^"]+)"\)', code):
        assert os.path.exists(os.path.join(ROOT, "notebooks", "Data", match)), match


def test_readme_lists_every_notebook():
    with open(os.path.join(REFERENCE_DIR, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    for path in NOTEBOOKS:
        assert os.path.basename(path) in readme, os.path.basename(path) + " missing from README"

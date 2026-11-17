import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def repo():
    return REPO


@pytest.fixture
def make_notebook(tmp_path):
    def _make(name, cells):
        nb = {"cells": [{"cell_type": t, "source": s, "metadata": {}, **({"outputs": [], "execution_count": None} if t == "code" else {})}
                        for t, s in cells], "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
        p = tmp_path / "notebooks" / (name + ".ipynb")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(nb), encoding="utf-8")
        return p
    return _make

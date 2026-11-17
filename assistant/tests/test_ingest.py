from assistant.ingest import api_names, build_pieces, chunk_notebook, chunk_source, load_index, save_index


def test_notebook_pieces_keep_location_and_split_at_headings(make_notebook, tmp_path):
    p = make_notebook("Demo", [
        ("markdown", "# Demo\nIntro text."),
        ("code", "import optimus"),
        ("markdown", "## Define the source"),
        ("code", "src = optimus.source.create_bowl(500e3)"),
    ])
    pieces = chunk_notebook(p, tmp_path)
    assert len(pieces) == 2
    first, second = pieces
    assert first.source == "notebooks/Demo.ipynb"
    assert first.location == "cells 0-1"
    assert second.heading == "Define the source"
    assert "create_bowl" in second.text
    assert second.citation() == "notebooks/Demo.ipynb, cells 2-3 (Define the source)"


def test_large_code_cell_is_never_split(make_notebook, tmp_path):
    code = "\n".join("x%d = %d" % (i, i) for i in range(400))
    p = make_notebook("Big", [("markdown", "intro"), ("code", code), ("markdown", "after")])
    pieces = chunk_notebook(p, tmp_path, target_tokens=50)
    assert any(code in piece.text for piece in pieces)


def test_outputs_are_dropped(make_notebook, tmp_path):
    p = make_notebook("Out", [("code", "print(1)")])
    pieces = chunk_notebook(p, tmp_path)
    assert "outputs" not in pieces[0].text


def test_source_pieces_and_api_names(tmp_path):
    pkg = tmp_path / "pkg" / "sub"
    pkg.mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("from . import sub\nglobal_value = 3\n")
    (pkg / "__init__.py").write_text("from .mod import make_thing\n")
    (pkg / "mod.py").write_text(
        'def make_thing(size, label="a"):\n    """Make a thing."""\n    return size\n\n'
        "def _private():\n    pass\n\nclass Thing:\n    def run(self):\n        pass\n"
    )
    pieces = chunk_source(pkg / "mod.py", tmp_path)
    assert [p.heading for p in pieces] == ["pkg.sub.mod.make_thing", "pkg.sub.mod.Thing"]
    assert "Make a thing." in pieces[0].text and pieces[0].location == "lines 1-3"

    names = api_names(tmp_path, "pkg")
    assert names["pkg.sub"] == "module"
    assert names["pkg.sub.make_thing"] == "name"  # re-exported in __init__
    assert names["pkg.sub.mod.make_thing"] == "function"
    assert names["pkg.sub.mod.Thing"] == "class"
    assert names["pkg.global_value"] == "name"
    assert "pkg.sub.mod._private" not in names


def test_real_repository_is_indexed(repo, tmp_path):
    pieces = build_pieces(repo)
    notebooks = {p.source for p in pieces if p.kind == "notebook"}
    assert len(notebooks) == len(list((repo / "notebooks").glob("*.ipynb")))
    assert any(p.heading == "optimus.source.bowl.create_bowl" for p in pieces)
    assert len({p.id for p in pieces}) == len(pieces), "piece ids must be unique"
    save_index(pieces, tmp_path / "i.json", "abc")
    assert load_index(tmp_path / "i.json") == pieces


def test_real_api_names(repo):
    names = api_names(repo)
    assert names["optimus.source.create_bowl"] == "name"
    assert names["optimus.source.bowl.create_bowl"] == "function"
    assert names["optimus.material.load_material"] == "name"
    assert "optimus.global_parameters" in names

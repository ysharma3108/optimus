from assistant import config
from assistant.checks import (ERROR, INFO, WARNING, check_api_names, check_mesh, check_syntax,
                              elements_per_wavelength, has_errors, mesh_verdict, run_all)
from assistant.ingest import api_names


def test_syntax_error_is_reported_with_line():
    issues = check_syntax("x = (1,\n")
    assert issues and issues[0].level == ERROR and issues[0].line >= 1
    assert check_syntax("x = 1") == []


def test_api_names_against_the_real_source(repo):
    api = api_names(repo)
    code = (
        "import optimus\nimport optimus as op\nfrom optimus.source import create_bowl\n"
        "m = optimus.material.load_material('water')\n"
        "s = create_bowl(1)\n"
        "op.global_parameters.linalg.tol = 1e-6\n"
        "x = optimus.source.make_magic_bowl()\n"
        "from optimus.source import nope\n"
    )
    issues = check_api_names(code, api)
    bad = sorted(i.message for i in issues)
    assert bad == ["'optimus.source.make_magic_bowl' does not exist in OptimUS",
                   "'optimus.source.nope' does not exist in OptimUS"]


def test_elements_per_wavelength_formula():
    # water, 500 kHz: wavelength = 1482.3 / 5e5 = 2.9646 mm
    assert elements_per_wavelength(5e5, 2.9646e-3 / 6, 1482.3) == __import__("pytest").approx(6.0)


def test_mesh_verdict_thresholds():
    assert mesh_verdict(config.ELEMENTS_PER_WAVELENGTH_FAIL - 0.1) == ERROR
    assert mesh_verdict(5.0) == WARNING
    assert mesh_verdict(config.ELEMENTS_PER_WAVELENGTH_WARN) == INFO


def test_mesh_check_reads_numbers_from_the_script():
    code = "frequency = 500e3\nwavelength = 1482.3 / frequency\nelement_size = wavelength / 3\n"
    issues = check_mesh(code)
    assert len(issues) == 1 and issues[0].level == ERROR and "3.0 elements" in issues[0].message


def test_mesh_check_uses_the_slowest_material():
    speeds = {"water": 1482.3, "fat": 1440.0}
    code = ("import optimus\nm = optimus.material.load_material(['water', 'fat'])\n"
            "frequency = 1e6\nelement_size = 1440.0 / 1e6 / 6\n")
    issues = check_mesh(code, speeds)
    assert issues[0].level == INFO and "c = 1440" in issues[0].message


def test_mesh_check_says_when_it_cannot_check():
    code = "frequency = 1e6\ng = make(element_size=compute())\n"
    assert check_mesh(code)[0].level == INFO
    assert "run time" in check_mesh(code)[0].message


def test_run_all_stops_at_syntax_errors():
    assert has_errors(run_all("def (:"))
    assert not has_errors(run_all("frequency = 1e3\nelement_size = 0.1\n"))

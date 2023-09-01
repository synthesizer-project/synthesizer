import os
import pathlib
import runpy
import pytest

scripts = pathlib.Path(__file__).parents[1]
scripts = scripts / "examples"
scripts = scripts.rglob("*.py")


@pytest.mark.parametrize('script', scripts)
def test_script_execution(script):
    import matplotlib as mpl
    mpl.use('template')
    runpy.run_path(str(script))

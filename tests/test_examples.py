import os
import pathlib
import runpy
import pytest

scripts = pathlib.Path(__file__).parents[1]
scripts = scripts / "examples"
scripts = scripts.rglob("*.py")

# for p in scripts.rglob("*.py"):
#     print(p)

# for path in scripts:
#     print(path)

@pytest.mark.parametrize('script', scripts)
def test_script_execution(script):
    runpy.run_path(str(script))

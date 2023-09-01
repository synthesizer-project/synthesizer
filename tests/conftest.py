# conftest.py
import pytest

def pytest_collectreport(report):
    print("TESTS CONFTEST loaded")

@pytest.fixture
def currpath(request):
    return str(request.node.fspath)

def pytest_configure():
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    print("TESTS pytest_configure")
    import matplotlib as mpl
    mpl.use('template')

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    print("TESTS pytest_sessionstart")
    import matplotlib as mpl
    mpl.use('template')

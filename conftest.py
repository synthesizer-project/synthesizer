def pytest_configure():
    """ called before ``pytest_runtest_call(item). """
    import matplotlib as mpl
    mpl.use('template')  

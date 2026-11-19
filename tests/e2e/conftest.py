def pytest_configure(config):
    config.addinivalue_line(
        "markers", "e2e: browser test of the built web app against a real API + test DB "
                   "(about 20 s; deselect with -m 'not e2e')")

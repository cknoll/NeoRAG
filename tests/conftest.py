"""
This file is optional. It serves to configure pytest. In particular it offers the option to use the ipydex
excepthook. This is a mechanism which opens an interactive ipython shell in the context where an exception
occurs.

This might be very helpful for debugging. However, as this can confuse unprepared users and als is not
helpful e.g. during continuous integration runs it is deactivated by default. To activate set the
appropriate environment variable to "True" via `export PYTEST_IPS=True`.

If you do not need this feature, you can safely delete this file.
"""

import os
import pytest

# use `export PYTEST_IPS=True` to activate this

if os.getenv("PYTEST_IPS") == "True":
    import ipydex

    def pytest_runtest_setup(item):
        print("This invocation of pytest is customized")

    def pytest_exception_interact(node, call, report):
        ipydex.ips_excepthook(call.excinfo.type, call.excinfo.value, call.excinfo.tb, leave_ut=True)

def pytest_collection_modifyitems(config, items):
    """Skip tests marked ``slow`` unless ``--run-slow`` is passed.

    Slow tests typically pull in heavy ML stacks (HuggingFace, Qdrant) or
    perform real I/O against a model index, so they are opt-in by default.
    """
    if config.getoption("--run-slow"):
        return
    skip_slow = __import__("pytest").mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: mark test as slow (skipped unless --run-slow is given)"
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--allow-api-calls",
        action="store_true",
        default=False,
        help="Allow tests that make real API calls",
    )


    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow integration tests (real retriever, model downloads).",
    )


@pytest.fixture
def allow_api_calls(request):
    """Fixture to check if --allow-api-calls flag is set."""
    return request.config.getoption("--allow-api-calls")

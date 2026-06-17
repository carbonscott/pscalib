"""Shared pytest fixtures for the pscalib acceptance suite.

The cross-check tests are dual-mode: their ``__main__`` runners hand each test a
writable scratch directory (``out_dir=tmp``) for snapshot/render artifacts, while
under pytest the same parameter is supplied by this fixture. Without it, every
``out_dir``-taking test errors at setup with "fixture 'out_dir' not found".
"""

import pytest


@pytest.fixture
def out_dir(tmp_path):
    """Writable output directory (path string) for snapshot/render artifacts.

    Mirrors the ``out_dir=tmp`` the tests' ``__main__`` blocks pass themselves;
    pytest hands each test its own ``tmp_path`` instead.
    """
    return str(tmp_path)

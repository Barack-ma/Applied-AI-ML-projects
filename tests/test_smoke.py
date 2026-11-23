"""Minimal smoke tests so pytest has something to collect."""

from applied_ml import __version__


def test_package_version_is_string() -> None:
    assert isinstance(__version__, str)
    assert __version__

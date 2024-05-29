from __future__ import annotations

import importlib.metadata

import HD_BET as m


def test_version():
    assert importlib.metadata.version("HD_BET") == m.__version__

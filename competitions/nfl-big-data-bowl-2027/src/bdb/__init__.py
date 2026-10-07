"""Helper package for the NFL Big Data Bowl 2027 workspace.

Usage from a notebook or script (adds src/ to sys.path if needed):

    import sys; sys.path.insert(0, "../src")   # from notebooks/
    from bdb import load, paths

Everything reads from ``paths.DATA`` (``data/`` next to this package's project root) and prefers
the parquet cache built by ``scripts/setup_data.py --parquet`` when it exists.
"""

from . import load, paths, tracking  # noqa: F401

__all__ = ["load", "paths", "tracking"]
